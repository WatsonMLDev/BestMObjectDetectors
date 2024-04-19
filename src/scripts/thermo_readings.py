import RPi.GPIO as GPIO
import threading
import time
import csv
import os

class MAX6675(object):
    '''Python driver for [MAX6675 Cold-Junction Compensated Thermocouple-to-Digital Converter](http://www.adafruit.com/datasheets/MAX6675.pdf)
     Requires:
     - The [GPIO Library](https://code.google.com/p/raspberry-gpio-python/) (Already on most Raspberry Pi OS builds)
     - A [Raspberry Pi](http://www.raspberrypi.org/)
    '''
    def __init__(self, cs_pin, clock_pin, data_pin, units = "c", board = GPIO.BCM):
        '''Initialize Soft (Bitbang) SPI bus
        Parameters:
        - cs_pin:    Chip Select (CS) / Slave Select (SS) pin (Any GPIO)
        - clock_pin: Clock (SCLK / SCK) pin (Any GPIO)
        - data_pin:  Data input (SO / MOSI) pin (Any GPIO)
        - units:     (optional) unit of measurement to return. ("c" (default) | "k" | "f")
        - board:     (optional) pin numbering method as per RPi.GPIO library (GPIO.BCM (default) | GPIO.BOARD)
        '''
        self.cs_pin = cs_pin
        self.clock_pin = clock_pin
        self.data_pin = data_pin
        self.units = units
        self.data = None
        self.board = board

        # Initialize needed GPIO
        GPIO.setmode(self.board)
        GPIO.setup(self.cs_pin, GPIO.OUT)
        GPIO.setup(self.clock_pin, GPIO.OUT)
        GPIO.setup(self.data_pin, GPIO.IN)

        # Pull chip select high to make chip inactive
        GPIO.output(self.cs_pin, GPIO.HIGH)

    def get(self):
        '''Reads SPI bus and returns current value of thermocouple.'''
        self.read()
        self.checkErrors()
        return getattr(self, "to_" + self.units)(self.data_to_tc_temperature())

    def read(self):
        '''Reads 16 bits of the SPI bus & stores as an integer in self.data.'''
        bytesin = 0
        # Select the chip
        GPIO.output(self.cs_pin, GPIO.LOW)
        # Read in 16 bits
        for i in range(16):
            GPIO.output(self.clock_pin, GPIO.LOW)
            time.sleep(0.001)
            bytesin = bytesin << 1
            if (GPIO.input(self.data_pin)):
                bytesin = bytesin | 1
            GPIO.output(self.clock_pin, GPIO.HIGH)
        time.sleep(0.001)
        # Unselect the chip
        GPIO.output(self.cs_pin, GPIO.HIGH)
        # Save data
        self.data = bytesin

    def checkErrors(self, data_16 = None):
        '''Checks errors on bit D2'''
        if data_16 is None:
            data_16 = self.data
        noConnection = (data_16 & 0x4) != 0       # tc input bit, D2

        if noConnection:
            raise MAX6675Error("No Connection") # open thermocouple

    def data_to_tc_temperature(self, data_16 = None):
        '''Takes an integer and returns a thermocouple temperature in celsius.'''
        if data_16 is None:
            data_16 = self.data
        # Remove bits D0-3
        tc_data = ((data_16 >> 3) & 0xFFF)
        # 12-bit resolution
        return (tc_data * 0.25)

    def to_c(self, celsius):
        '''Celsius passthrough for generic to_* method.'''
        return celsius

    def to_k(self, celsius):
        '''Convert celsius to kelvin.'''
        return celsius + 273.15

    def to_f(self, celsius):
        '''Convert celsius to fahrenheit.'''
        return celsius * 9.0/5.0 + 32

    def cleanup(self):
        '''Selective GPIO cleanup'''
        GPIO.setup(self.cs_pin, GPIO.IN)
        GPIO.setup(self.clock_pin, GPIO.IN)


class MAX6675Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)



class TemperatureReader:
    def __init__(self, cs_pin, sck, data_pin, units="c", soc_temp_path="/sys/class/thermal/thermal_zone0/temp"):
        self.thermocouple = MAX6675(cs_pin, sck, data_pin, units)
        self.soc_temp_path = soc_temp_path
        self.running = False
        self.ambient_temps = []  # Stores all ambient temperature readings
        self.soc_temps = []  # Stores all SoC temperature readings

    def get_soc_temperature(self):
        # Read the SoC temperature from the system file
        try:
            with open(self.soc_temp_path, 'r') as f:
                temp_str = f.read().strip()  # Read temperature string
                return float(temp_str) / 1000.0  # Convert to Celsius
        except IOError:
            print("Failed to read SoC temperature.")
            return None

    def start(self):
        self.running = True
        thread = threading.Thread(target=self.record_temperature)
        thread.daemon = True  # Daemonize thread
        thread.start()

    def record_temperature(self):
        print("Starting temperature recording.")
        while self.running:
            ambient_temp = self.thermocouple.get()
            self.ambient_temps.append(ambient_temp)

            soc_temp = self.get_soc_temperature()
            self.soc_temps.append((time.time(), soc_temp))  # Save time and SoC temperature

            # Print temperatures
            # print(f"Ambient: {ambient_temp} {self.thermocouple.units}, SoC: {soc_temp} C")

            # Sleep for some time between readings
            time.sleep(1)

    def stop(self, model_name):
        self.running = False
        self.thermocouple.cleanup()

        # Generate a new CSV filename if one already exists
        file_number = 0
        while True:
            filename = f'results/soc_{model_name}_temperatures{"" if file_number == 0 else "_" + str(file_number)}.csv'
            if not os.path.isfile(filename):
                break
            file_number += 1

        # Write SoC temperature readings to the CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'SoC Temperature (C)'])
            writer.writerows(self.soc_temps)

    @property
    def average_ambient_temperature(self):
        return sum(self.ambient_temps) / len(self.ambient_temps) if self.ambient_temps else None

    @property
    def average_soc_temperature(self):
        return sum(temp for _, temp in self.soc_temps) / len(self.soc_temps) if self.soc_temps else None

