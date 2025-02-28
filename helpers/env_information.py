# prints information regarding available Maps and Weather Presets. Carla Server has to be running
import carla

CARLA_HOST = 'localhost'
CARLA_PORT = 2000
    
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(5.0)  # seconds

# Get the world object
world = client.get_world()

# Print all available maps
available_maps = client.get_available_maps()
print("Available Maps:")
for map_name in available_maps:
    print(map_name)

# Print all available weather presets
weather_presets = [x for x in dir(carla.WeatherParameters) if not x.startswith('__')]
print("\nAvailable Weather Presets:")
for preset_name in weather_presets:
    print(preset_name)