import odatse

def initialize():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Data-analysis software of quantum beam "
            "diffraction experiments for 2D material structure"
        )
    )
    parser.add_argument("inputfile", help="input file with TOML format")
    parser.add_argument("--version", action="version", version=odatse.__version__)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--init", action="store_true", help="initial start (default)")
    mode_group.add_argument("--resume", action="store_true", help="resume intterupted run")
    mode_group.add_argument("--cont", action="store_true", help="continue from previous run")

    parser.add_argument("--reset_rand", action="store_true", default=False, help="new random number series in resume or continue mode")

    args = parser.parse_args()


    if args.init is True:
        run_mode = "initial"
    elif args.resume is True:
        run_mode = "resume"
        if args.reset_rand is True:
            run_mode = "resume-resetrand"
    elif args.cont is True:
        run_mode = "continue"
        if args.reset_rand is True:
            run_mode = "continue-resetrand"
    else:
        run_mode = "initial"  # default

    info = odatse.Info.from_file(args.inputfile)
    # info.algorithm.update({"run_mode": run_mode})

    return info, run_mode
