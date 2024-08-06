from containers._meta import _ConfigClass


class MonitoringSystemConfig(_ConfigClass):
    monitor_cpu: bool = False
    monitor_ram: bool = False


    def set_all(self, value:bool) -> None:
        self.monitor_cpu = value
        self.monitor_ram = value

    def __repr__(self) -> str:
        return super().__repr__()
    


