def unicode_utf8_bytes_to_str_wrong(bytestring: bytes) -> str:
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

if __name__ == "__main__":
    print(unicode_utf8_bytes_to_str_wrong("🫠".encode("utf-8")))