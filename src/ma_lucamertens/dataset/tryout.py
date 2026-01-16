if __name__ == "__main__":
    import platform

    import tartanair as ta

    _U_ARCH = platform.uname().machine

    print(f"Current architecture: {_U_ARCH}")

    tartanair_data_root = "~/Development/tartanair-v2"
    ta.init(tartanair_data_root)
