"""Run the released scalar analysis, with optional figure reproduction."""
from pathlib import Path
import argparse,subprocess,sys
from verify_release import verify

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--figures',action='store_true',help='Also check public masks and regenerate all five figures')
    args=ap.parse_args()
    verify()
    subprocess.run([sys.executable,str(ROOT/'software/run_reproduce.py'),'--output',str(ROOT/'results/statistics')],check=True)
    if args.figures:
        for name in ['plot_figures_1_to_4.py','plot_figure_5.py']:
            subprocess.run([sys.executable,str(ROOT/'scripts'/name)],check=True)
    print('Completed. Outputs are in results/.')

if __name__=='__main__':main()
