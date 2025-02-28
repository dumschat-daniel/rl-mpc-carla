import carla
import time
import random

# shows all available vehicle blueprints
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
    
client = carla.Client(CARLA_HOST, CARLA_PORT)
client.set_timeout(5.0)  # seconds
world = client.get_world()

blueprint_library = world.get_blueprint_library()
vehicle_blueprints = blueprint_library.filter('vehicle.*')
spawn_point = world.get_map().get_spawn_points()[0]

spectator = world.get_spectator()
spectator.set_transform(spawn_point)

for blueprint in vehicle_blueprints:
    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)
    actor = world.spawn_actor(blueprint, spawn_point)
    print(f"Spawned {actor.type_id}")
    time.sleep(2)
    actor.destroy()

