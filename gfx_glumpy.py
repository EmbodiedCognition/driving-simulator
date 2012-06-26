'''This file contains code for displaying the simulation graphically.'''

import glumpy
import math
import OpenGL.GL as gl
import OpenGL.GLU as glu
import sys

TAU = 2 * math.pi

REALITY_ALPHA = 0.3
ESTIMATE_ALPHA = 0.7


def draw_cone(color, pos, vel, speed):
    gl.glColor(*color)

    gl.glPushMatrix()

    x, y = pos
    gl.glTranslate(x, y, 3)

    vx, vy = vel
    gl.glRotate(360 / TAU * math.atan2(vy, vx), 0, 0, 1)
    gl.glRotate(90, 0, 0, 1)
    gl.glRotate(90, 1, 0, 0)

    q = glu.gluNewQuadric()
    glu.gluCylinder(q, 2, 0, max(1, speed), 10, 10)
    glu.gluDeleteQuadric(q)

    gl.glPopMatrix()


def draw_sphere(color, pos):
    gl.glColor(*color)

    gl.glPushMatrix()

    x, y = pos
    gl.glTranslate(x, y, 3)

    q = glu.gluNewQuadric()
    glu.gluSphere(q, 2, 10, 10)
    glu.gluDeleteQuadric(q)

    gl.glPopMatrix()


elapsed = 0

def main(simulator):
    fig = glumpy.figure()
    world = fig.add_figure()

    @fig.event
    def on_draw():
        fig.clear()

        # draw the opengl driving world.

        w = world.width
        h = world.height
        x = world.x
        y = world.y

        gl.glBegin(gl.GL_QUADS)
        gl.glColor(0.2, 0.2, 0.2)
        gl.glNormal(0, 0, 1)
        gl.glVertex(x, y, 0)
        gl.glVertex(x + w, y, 0)
        gl.glVertex(x + w, y + h, 0)
        gl.glVertex(x, y + h, 0)
        gl.glEnd()

        z = min(w, h) / 250.
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glTranslate(x + w / 2., y + h / 2., 10)
        gl.glScale(z, z, 1)
        #gl.glTranslate(-leader.position[0], -leader.position[1], 0)

        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, (0, 0, 100, 1))

        # draw lanes.
        gl.glLineWidth(2)
        gl.glColor(0, 0, 0)
        for track in simulator.tracks:
            gl.glBegin(gl.GL_LINE_STRIP)
            for a, b in track:
                gl.glVertex(a, b, 1)
            gl.glEnd()

        # draw cars.
        simulator.agent.draw(sys.modules[__name__], 1, 1, 0, REALITY_ALPHA)
        simulator.leader.draw(sys.modules[__name__], 1, 0, 0, REALITY_ALPHA)

        gl.glPopMatrix()

    @fig.event
    def on_idle(dt):
        global elapsed
        elapsed += dt
        while elapsed > simulator.dt:
            elapsed -= simulator.dt
            simulator.step()
            fig.redraw()

    @fig.event
    def on_key_press(key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        elif key == glumpy.window.key.SPACE:
            global ESTIMATE_ALPHA, REALITY_ALPHA
            ESTIMATE_ALPHA, REALITY_ALPHA = REALITY_ALPHA, ESTIMATE_ALPHA
        else:
            simulator.reset()
        fig.redraw()

    glumpy.show()
