from dataclasses import dataclass, fields


@dataclass
class Expectations():
    x : float
    y : float
    z : float

    def __repr__(self) -> str:        
        s = f"{self.__class__.__name__}:"
        for f in fields(self):
            s += "\n"
            value = getattr(self, f.name)
            s += f"  {f.name}: {value}"
        return s
    

@dataclass
class Measurements():
    expectations : Expectations
    energy : float

    def __repr__(self) -> str:        
        s = f"{self.__class__.__name__}:"
        for f in fields(self):
            s += "\n"
            value = getattr(self, f.name)
            s += f"{f.name}: {value}"
        return s