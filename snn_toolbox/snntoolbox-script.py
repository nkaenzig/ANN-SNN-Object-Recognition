#!C:\Users\Nicolas\Anaconda3\envs\envname\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'snntoolbox==0.1.2','console_scripts','snntoolbox'
__requires__ = 'snntoolbox==0.1.2'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('snntoolbox==0.1.2', 'console_scripts', 'snntoolbox')()
    )
