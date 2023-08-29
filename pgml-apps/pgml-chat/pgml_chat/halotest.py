import time
from halo import Halo

with Halo(text="Testing Halo for 5 seconds", spinner="dots"):
    time.sleep(5)

