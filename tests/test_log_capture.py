import sys

from weclone.utils.log import capture_output


def test_capture_output_preserves_standard_stream_interface() -> None:
    original_stdout = sys.stdout
    observed = {}

    @capture_output
    def inspect_stream() -> None:
        observed["isatty"] = sys.stdout.isatty()
        observed["fileno"] = sys.stdout.fileno()
        observed["encoding"] = sys.stdout.encoding

    inspect_stream()

    assert observed == {
        "isatty": original_stdout.isatty(),
        "fileno": original_stdout.fileno(),
        "encoding": original_stdout.encoding,
    }
