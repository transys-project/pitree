import pensieve
import robustmpc
import hotdash
import pensiedt
import robustmdt
import hotdadt
import argparse


def main(args):
    if args.dt is not None:
        if args.abr == 'pensieve':
            pensiedt.PensieveDT().main(args)
        elif args.abr == 'robustmpc':
            robustmdt.RobustMPCDT().main(args)
        elif args.abr == 'hotdash':
            hotdadt.HotdashDT().main(args)
    else:
        if args.abr == 'pensieve':
            pensieve.Pensieve().main(args)
        elif args.abr == 'robustmpc':
            robustmpc.RobustMPC().main(args)
        elif args.abr == 'hotdash':
            hotdash.Hotdash().main(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--abr', metavar='ABR', choices=['pensieve', 'robustmpc', 'hotdash'])
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-d', '--dt', default=None)
    parser.add_argument('-q', '--qoe-metric', choices=['lin', 'log', 'hd'])
    parser.add_argument('-t', '--traces', choices=['norway', 'fcc', 'oboe', 'fcc18', 'ghent', 'hsr', 'surrey'])

    args = parser.parse_args()
    main(args)
