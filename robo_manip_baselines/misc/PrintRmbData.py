import argparse

from robo_manip_baselines.common import RmbData


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("path", type=str, help="path to data (*.hdf5 or *.rmb)")

    return parser.parse_args()


class PrintRmbData:
    def __init__(self, path):
        self.path = path

    def run(self):
        print(f"[{self.__class__.__name__}] Open {self.path}")

        with RmbData(self.path) as rmb_data:
            print(f"[{self.__class__.__name__}] Attrs:")
            for k, v in rmb_data.attrs.items():
                print(f"  - {k}: {v}")

            print(f"[{self.__class__.__name__}] Data:")
            for k in rmb_data.keys():
                v = rmb_data[k]
                print(f"  - {k}: {v.shape}")


if __name__ == "__main__":
    print_data = PrintRmbData(**vars(parse_argument()))
    print_data.run()
