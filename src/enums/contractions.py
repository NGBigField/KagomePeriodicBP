from enum import Enum, auto

class ContractionDepth(Enum):
    Full = auto()
    ToMessage = auto()
    ToCore = auto()

class ReduceToCoreMethod(Enum):
    EachDirectionToCore = auto()
    DoubleMPSZipping    = auto()

    @staticmethod
    def default()->"ReduceToCoreMethod":
        return ReduceToCoreMethod.DoubleMPSZipping


class ReduceToEdgeMethod(Enum):
    EachDirectionToCore = auto()
    EachDirectionToEdge = auto()
    DoubleMPSZipping    = auto()

    @staticmethod
    def from_int(x:int)->"ReduceToEdgeMethod":
        match x:
            case 0: return ReduceToEdgeMethod.EachDirectionToCore
            case 1: return ReduceToEdgeMethod.EachDirectionToEdge
            case 2: return ReduceToEdgeMethod.DoubleMPSZipping
            case _:
                raise ValueError("Not a possible int")
            
    @staticmethod
    def default()->"ReduceToEdgeMethod":
        return ReduceToEdgeMethod.DoubleMPSZipping
    
    def derived_reduction2core(self)->"ReduceToCoreMethod":
        match self:
            case ReduceToEdgeMethod.EachDirectionToCore : return ReduceToCoreMethod.EachDirectionToCore
            case ReduceToEdgeMethod.EachDirectionToEdge : return ReduceToCoreMethod.EachDirectionToCore
            case ReduceToEdgeMethod.DoubleMPSZipping    : return ReduceToCoreMethod.DoubleMPSZipping
            case _:
                raise ValueError("Not a possible int")
