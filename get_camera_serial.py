import pyzed.sl as sl

    
def get_connected_devices_serial():

    try:
        serials = sl.Camera.get_device_list()
    except NameError:
        return []
    serials = [str(serial.serial_number) for serial in serials]
    serials = sorted(serials)
    print("Connected ZED camera serials:", serials)
    return serials

if __name__ == "__main__":
    print(get_connected_devices_serial())