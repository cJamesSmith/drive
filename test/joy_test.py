import pygame
import time

if __name__ == "__main__":
    pygame.init()
    pygame.joystick.init()
    joystick_count = pygame.joystick.get_count()
    print(joystick_count)
    if joystick_count > 1:
        raise ValueError("Please Connect Just One Joystick")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    for _ in range(1000):
        time.sleep(1 / 25)
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                print(joystick.get_axis(0))
            elif event.type == pygame.JOYBUTTONDOWN:
                if joystick.get_button(0):
                    print(joystick.get_button(0))
                elif joystick.get_button(1):
                    print(joystick.get_button(1))
