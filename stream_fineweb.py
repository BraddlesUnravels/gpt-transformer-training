import argparse
import errno
import os
import queue
import signal
import sys
import time
import multiprocessing as mp
from pathlib import Path

import httpx
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url, set_client_factory


DEFAULT_DATASET_NAME = "HuggingFaceFW/fineweb"
DEFAULT_DATASET_CONFIG_NAME = "sample-10BT"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"

MAX_DOCS = 10_000

STREAM_RETRY_ATTEMPTS = 3
STREAM_RETRY_DELAY_SECONDS = 1
MAX_STREAM_RETRY_DELAY_SECONDS = 4

# If the child process makes no progress for this many seconds, kill it and skip file.
FILE_IDLE_TIMEOUT_SECONDS = 120

HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 15
HF_HUB_ETAG_TIMEOUT_SECONDS = 5
HF_HUB_HTTP_CONNECT_TIMEOUT_SECONDS = 5
HF_HUB_HTTP_READ_TIMEOUT_SECONDS = 15
HF_HUB_HTTP_WRITE_TIMEOUT_SECONDS = 15
HF_HUB_HTTP_POOL_TIMEOUT_SECONDS = 15


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--config", default=DEFAULT_DATASET_CONFIG_NAME)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
    parser.add_argument("--max-samples", type=int, default=MAX_DOCS)
    parser.add_argument("--file-idle-timeout", type=int, default=FILE_IDLE_TIMEOUT_SECONDS)

    # Example:
    # --file-prefix data/CC-MAIN-2025-26/
    parser.add_argument("--file-prefix", default=None)

    return parser.parse_args()


def configure_hf_hub_timeouts():
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_SECONDS))

    # Useful diagnostic / stability setting.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def create_hf_http_client():
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(
            connect=HF_HUB_HTTP_CONNECT_TIMEOUT_SECONDS,
            read=HF_HUB_HTTP_READ_TIMEOUT_SECONDS,
            write=HF_HUB_HTTP_WRITE_TIMEOUT_SECONDS,
            pool=HF_HUB_HTTP_POOL_TIMEOUT_SECONDS,
        ),
    )


def configure_hf_hub_http_client():
    set_client_factory(create_hf_http_client)


def read_hf_token():
    token = os.getenv("HF_TOKEN")

    if token:
        return token

    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]

    seen_paths = set()

    for path in candidate_paths:
        resolved_path = path.resolve()

        if resolved_path in seen_paths:
            continue

        seen_paths.add(resolved_path)

        if not path.is_file():
            continue

        with path.open("r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()

                if not line or line.startswith("#"):
                    continue

                if line.startswith("export "):
                    line = line[len("export "):].strip()

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)

                if key.strip() != "HF_TOKEN":
                    continue

                return value.strip().strip('"').strip("'")

    return None


def infer_file_prefix(config_name):
    """
    FineWeb has multiple directory patterns depending on the config/dump.

    If this does not find files for your config, pass --file-prefix manually.
    Example:
      --file-prefix data/CC-MAIN-2025-26/
    """
    if config_name == "sample-10BT":
        return "sample/10BT/"

    if config_name == "sample-100BT":
        return "sample/100BT/"

    if config_name == "sample-350BT":
        return "sample/350BT/"

    if config_name.startswith("CC-MAIN-"):
        return f"data/{config_name}/"

    return None


def list_parquet_files(dataset_name, config_name, token, file_prefix=None):
    prefix = file_prefix or infer_file_prefix(config_name)

    if prefix is None:
        raise ValueError(
            "Could not infer Parquet file prefix. Pass one explicitly, e.g. "
            "--file-prefix data/CC-MAIN-2025-26/"
        )

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset")

    parquet_files = [
        file_path
        for file_path in files
        if file_path.startswith(prefix) and file_path.endswith(".parquet")
    ]

    if not parquet_files:
        raise ValueError(
            f"No Parquet files found under prefix '{prefix}' in dataset '{dataset_name}'. "
            "Try passing --file-prefix manually."
        )

    return sorted(parquet_files)


def is_retryable_stream_error(error):
    if isinstance(error, OSError):
        retryable_errno_values = {
            errno.EBADF,
            errno.ECONNRESET,
            errno.EPIPE,
            errno.ETIMEDOUT,
        }

        if error.errno in retryable_errno_values:
            return True

    lowered_message = str(error).lower()

    retryable_fragments = [
        "bad file descriptor",
        "connection aborted",
        "connection reset",
        "timed out",
        "temporarily unavailable",
        "server disconnected",
        "incomplete read",
        "connection broken",
    ]

    return any(fragment in lowered_message for fragment in retryable_fragments)


def get_text_from_row(row, text_field):
    if not isinstance(row, dict):
        raise ValueError(f"Expected dataset row to be dict, received {type(row).__name__}")

    if text_field not in row:
        available_fields = list(row.keys())
        raise ValueError(
            f"Field '{text_field}' was not found in row. Available fields: {available_fields}"
        )

    text = row[text_field]

    if isinstance(text, str):
        return text

    raise ValueError(f"Field '{text_field}' exists but is not a string")


def stream_file_worker(file_url, text_field, max_samples_for_file, token, progress_queue):
    """
    Child process.

    If load_dataset() freezes here, the parent process kills this whole process
    and moves to the next Parquet file.
    """
    configure_hf_hub_timeouts()
    configure_hf_hub_http_client()

    emitted_count = 0

    try:
        dataset = load_dataset(
            "parquet",
            data_files={"train": file_url},
            split="train",
            streaming=True,
            token=token,
        )

        progress_queue.put({"status": "loaded", "emitted_count": emitted_count})

        for row in dataset:
            text = get_text_from_row(row, text_field)

            if not text:
                continue

            if text.endswith("\n"):
                sys.stdout.write(text)
            else:
                sys.stdout.write(text + "\n")

            emitted_count += 1

            if emitted_count % 100 == 0:
                sys.stdout.flush()
                progress_queue.put({
                    "status": "progress",
                    "emitted_count": emitted_count,
                })

            if max_samples_for_file is not None and emitted_count >= max_samples_for_file:
                break

        sys.stdout.flush()

        progress_queue.put({
            "status": "done",
            "emitted_count": emitted_count,
        })

    except Exception as error:
        progress_queue.put({
            "status": "error",
            "emitted_count": emitted_count,
            "error": repr(error),
            "retryable": is_retryable_stream_error(error),
        })


def terminate_process(process):
    if not process.is_alive():
        return

    process.terminate()
    process.join(timeout=5)

    if process.is_alive():
        process.kill()
        process.join(timeout=5)


def stream_file_with_process_guard(
    file_url,
    file_path,
    text_field,
    max_samples_for_file,
    token,
    file_idle_timeout,
):
    """
    Parent process.

    Starts a child process for one file. If the child freezes inside load_dataset()
    or during streaming, the parent kills it and returns the number of samples
    emitted before failure.
    """
    progress_queue = mp.Queue()

    process = mp.Process(
        target=stream_file_worker,
        args=(
            file_url,
            text_field,
            max_samples_for_file,
            token,
            progress_queue,
        ),
    )

    process.start()

    last_progress_time = time.time()
    last_emitted_count = 0
    final_status = None
    final_error = None
    final_retryable = True

    while True:
        try:
            message = progress_queue.get(timeout=1)

            status = message.get("status")
            last_emitted_count = message.get("emitted_count", last_emitted_count)
            last_progress_time = time.time()

            if status == "done":
                final_status = "done"
                break

            if status == "error":
                final_status = "error"
                final_error = message.get("error")
                final_retryable = message.get("retryable", True)
                break

        except queue.Empty:
            pass

        if not process.is_alive():
            break

        idle_seconds = time.time() - last_progress_time

        if idle_seconds > file_idle_timeout:
            print(
                f"Skipping file after {file_idle_timeout}s with no progress: {file_path}",
                file=sys.stderr,
            )
            terminate_process(process)
            return last_emitted_count, True

    process.join(timeout=5)

    if process.is_alive():
        terminate_process(process)

    if final_status == "done":
        return last_emitted_count, False

    if final_status == "error":
        print(
            f"Error while streaming file: {file_path}\n"
            f"Error: {final_error}",
            file=sys.stderr,
        )

        if not final_retryable:
            raise RuntimeError(f"Non-retryable error in file {file_path}: {final_error}")

        return last_emitted_count, True

    if process.exitcode not in (0, None):
        print(
            f"Child process exited with code {process.exitcode} while reading: {file_path}",
            file=sys.stderr,
        )
        return last_emitted_count, True

    return last_emitted_count, False


def main():
    args = parse_args()

    configure_hf_hub_timeouts()
    configure_hf_hub_http_client()

    token = read_hf_token()

    parquet_files = list_parquet_files(
        dataset_name=args.dataset,
        config_name=args.config,
        token=token,
        file_prefix=args.file_prefix,
    )

    print(f"Found {len(parquet_files)} Parquet files to stream.", file=sys.stderr)

    emitted_sample_count = 0
    skipped_file_count = 0

    for file_index, file_path in enumerate(parquet_files, start=1):
        if args.max_samples is not None and emitted_sample_count >= args.max_samples:
            break

        file_url = hf_hub_url(
            repo_id=args.dataset,
            filename=file_path,
            repo_type="dataset",
        )

        remaining_samples = None

        if args.max_samples is not None:
            remaining_samples = args.max_samples - emitted_sample_count

        print(
            f"Streaming file {file_index}/{len(parquet_files)}: {file_path}",
            file=sys.stderr,
        )

        retry_count = 0

        while retry_count <= STREAM_RETRY_ATTEMPTS:
            emitted_from_file, should_retry_or_skip = stream_file_with_process_guard(
                file_url=file_url,
                file_path=file_path,
                text_field=args.text_field,
                max_samples_for_file=remaining_samples,
                token=token,
                file_idle_timeout=args.file_idle_timeout,
            )

            emitted_sample_count += emitted_from_file

            if args.max_samples is not None and emitted_sample_count >= args.max_samples:
                sys.stdout.flush()
                print(
                    f"Finished after emitting {emitted_sample_count} samples.",
                    file=sys.stderr,
                )
                return

            if not should_retry_or_skip:
                break

            retry_count += 1

            if retry_count > STREAM_RETRY_ATTEMPTS:
                skipped_file_count += 1
                print(
                    f"Skipping file after repeated failures: {file_path}",
                    file=sys.stderr,
                )
                break

            wait_seconds = min(
                STREAM_RETRY_DELAY_SECONDS * (2 ** (retry_count - 1)),
                MAX_STREAM_RETRY_DELAY_SECONDS,
            )

            print(
                f"Retrying file in {wait_seconds}s "
                f"[{retry_count}/{STREAM_RETRY_ATTEMPTS}]: {file_path}",
                file=sys.stderr,
            )

            time.sleep(wait_seconds)

    sys.stdout.flush()

    print(
        f"Done. Emitted {emitted_sample_count} samples. "
        f"Skipped {skipped_file_count} files.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    # Required for safe multiprocessing on macOS/Windows.
    mp.set_start_method("spawn", force=True)
    main()