"""This file defines all the data classes."""
from dataclasses import dataclass, field
import datetime

@dataclass
class Experiment:
    """Class for keeping track of an item in inventory."""
    now: datetime.datetime = datetime.datetime.now()
    run_id: str = field(init=False)
    today: str = field(init=False)
    time: str = field(init=False)

    def __post_init__(self):
        self.run_id = self.now.strftime('%Y%m%d%H%M%S')
        self.today = self.now.strftime('%Y%m%d')
        self.time = self.now.strftime("%H-%M-%S")
