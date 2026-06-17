import argparse
import errno
import os
import signal
import sys
import time
from pathlib import Path
import httpx

from datasets import load_dataset
from huggingface_hub import set_client_factory


DEFAULT_DATASET_NAME = "allenai/c4"
DEFAULT_DATASET_CONFIG_NAME = "en"
DEFAULT_SPLIT = "train"
DEFAULT_TEXT_FIELD = "text"
MAX_DOCS = 10_000
STREAM_RETRY_ATTEMPTS = 3
STREAM_RETRY_DELAY_SECONDS = 1
MAX_STREAM_RETRY_DELAY_SECONDS = 4
ROW_FETCH_TIMEOUT_SECONDS = 15
HF_HUB_DOWNLOAD_TIMEOUT_SECONDS = 15
HF_HUB_ETAG_TIMEOUT_SECONDS = 5
HF_HUB_HTTP_CONNECT_TIMEOUT_SECONDS = 5
HF_HUB_HTTP_READ_TIMEOUT_SECONDS = 15
HF_HUB_HTTP_WRITE_TIMEOUT_SECONDS = 15
HF_HUB_HTTP_POOL_TIMEOUT_SECONDS = 15


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
  parser.add_argument("--split", default=DEFAULT_SPLIT)
  parser.add_argument("--text-field", default=DEFAULT_TEXT_FIELD)
  parser.add_argument("--max-samples", type=int, default=MAX_DOCS)
  return parser.parse_args()


def get_text_from_row(row, text_field):
  if not isinstance(row, dict):
    raise ValueError(f"Expected dataset row to be a dict, received {type(row).__name__}")

  if text_field not in row:
    available_fields = list(row.keys())
    raise ValueError(
      f"Field '{text_field}' was not found in dataset row; available fields: {available_fields}"
    )

  text = row[text_field]

  if isinstance(text, str):
    return text

  raise ValueError(f"Field '{text_field}' exists but is not a string")


def is_retryable_stream_error(error):
  if isinstance(error, OSError):
    retryable_errno_values = {
      errno.EBADF,
      errno.ECONNRESET,
      errno.EPIPE,
      errno.ETIMEDOUT
    }

    if error.errno in retryable_errno_values:
      return True

  lowered_message = str(error).lower()
  retryable_fragments = [
    "bad file descriptor",
    "connection aborted",
    "connection reset",
    "timed out",
    "temporarily unavailable"
  ]

  return any(fragment in lowered_message for fragment in retryable_fragments)


def load_streaming_dataset(dataset_name, split, token):
  dataset_config_name = None

  if dataset_name == DEFAULT_DATASET_NAME:
    dataset_config_name = DEFAULT_DATASET_CONFIG_NAME

  if dataset_config_name:
    return load_dataset(
      dataset_name,
      dataset_config_name,
      split=split,
      streaming=True,
      token=token
    )

  return load_dataset(
    dataset_name,
    split=split,
    streaming=True,
    token=token
  )


class RowFetchTimeoutError(TimeoutError):
  pass


def raise_row_fetch_timeout(signum, frame):
  raise RowFetchTimeoutError(
    f"Timed out waiting for next streamed row after {ROW_FETCH_TIMEOUT_SECONDS} seconds"
  )


def read_next_row(dataset_iterator):
  if not hasattr(signal, "SIGALRM"):
    return next(dataset_iterator)

  previous_handler = signal.getsignal(signal.SIGALRM)
  signal.signal(signal.SIGALRM, raise_row_fetch_timeout)
  signal.setitimer(signal.ITIMER_REAL, ROW_FETCH_TIMEOUT_SECONDS)

  try:
    return next(dataset_iterator)
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous_handler)


def configure_hf_hub_timeouts():
  os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(HF_HUB_DOWNLOAD_TIMEOUT_SECONDS))
  os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(HF_HUB_ETAG_TIMEOUT_SECONDS))


def create_hf_http_client():
  return httpx.Client(
    follow_redirects=True,
    timeout=httpx.Timeout(
      connect=HF_HUB_HTTP_CONNECT_TIMEOUT_SECONDS,
      read=HF_HUB_HTTP_READ_TIMEOUT_SECONDS,
      write=HF_HUB_HTTP_WRITE_TIMEOUT_SECONDS,
      pool=HF_HUB_HTTP_POOL_TIMEOUT_SECONDS
    )
  )


def configure_hf_hub_http_client():
  set_client_factory(create_hf_http_client)


def read_hf_token():
  token = os.getenv("HF_TOKEN")

  if token:
    return token

  candidate_paths = [
    Path.cwd() / ".env",
    Path(__file__).resolve().parent / ".env"
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


def main():
  args = parse_args()
  configure_hf_hub_timeouts()
  configure_hf_hub_http_client()
  token = read_hf_token()
  emitted_sample_count = 0
  processed_row_count = 0
  retry_count = 0

  while args.max_samples is None or emitted_sample_count < args.max_samples:
    dataset = load_streaming_dataset(args.dataset, args.split, token)

    if processed_row_count > 0:
      dataset = dataset.skip(processed_row_count)

    try:
      dataset_iterator = iter(dataset)

      while True:
        row = read_next_row(dataset_iterator)
        processed_row_count += 1
        retry_count = 0
        text = get_text_from_row(row, args.text_field)

        if not text:
          continue

        if text.endswith("\n"):
          sys.stdout.write(text)
        else:
          sys.stdout.write(text + "\n")

        emitted_sample_count += 1

        if emitted_sample_count % 100 == 0:
          sys.stdout.flush()

        if args.max_samples is not None and emitted_sample_count >= args.max_samples:
          sys.stdout.flush()
          return

      break
    except StopIteration:
      break
    except Exception as error:
      if not is_retryable_stream_error(error):
        raise

      retry_count += 1

      if retry_count > STREAM_RETRY_ATTEMPTS:
        processed_row_count += 1
        retry_count = 0
        print(
          f"Skipping row offset {processed_row_count} after repeated "
          f"transient stream errors: {error}",
          file=sys.stderr
        )
        continue

      wait_seconds = min(
        STREAM_RETRY_DELAY_SECONDS * (2 ** (retry_count - 1)),
        MAX_STREAM_RETRY_DELAY_SECONDS
      )
      print(
        f"Transient stream error after {emitted_sample_count} emitted samples "
        f"(row offset {processed_row_count}): {error}. "
        f"Retrying in {wait_seconds}s [{retry_count}/{STREAM_RETRY_ATTEMPTS}]",
        file=sys.stderr
      )
      time.sleep(wait_seconds)

  sys.stdout.flush()


if __name__ == "__main__":
  main()
