from typing import List
from typing import Optional

from pydantic import BaseModel
class Item(BaseModel):
    name: str
    year: int
    selling_price: Optional[int] = None
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[str] = None
    engine: Optional[str] = None
    max_power: Optional[str] = None
    torque: Optional[str] = None
    seats: Optional[float] = None

class Items(BaseModel):
    objects: List[Item]
