import carla
import time

# setup time checks the time it takes for a vehicle to accept inputs after spawn
client = carla.Client('localhost', 2000)
client.set_timeout(5.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_point = world.get_map().get_spawn_points()[0]


try:
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0))

    start_time = time.time()

    
    while True:
        if vehicle.get_location().x != 0 and (vehicle.get_location().x != spawn_point.location.x or vehicle.get_location().y != spawn_point.location.y):
            break
    print(f"async mode: time taken {time.time() - start_time} seconds.")
finally:
    if vehicle is not None:
        vehicle.destroy()


settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.05
world.apply_settings(settings)

try:

    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

    world.tick()

    vehicle.apply_control(carla.VehicleControl(throttle=1.0))

    initial_location = vehicle.get_location()

    tick_count = 0

    while True:
        world.tick()
        tick_count += 1
        current_location = vehicle.get_location()
        if (current_location.x != initial_location.x or 
            current_location.y != initial_location.y):
            break

    print(f"sync mode: ticks taken {tick_count} which equals {tick_count * settings.fixed_delta_seconds} seconds.")
finally:
    if vehicle is not None:
        vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)

