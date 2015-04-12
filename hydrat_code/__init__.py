import sys

import configuration
import cli


# Global configuration
config = configuration.read_configuration()
configuration.load_configuration(config)

# Global random number generator
rng = configuration.rng 

def main():
  CLI = cli.HydratCmdln()
  sys.exit( CLI.main() )

if __name__ == "__main__":
  main()
