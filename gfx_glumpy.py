'''This file contains code for displaying the simulation graphically.'''

import glumpy
import math
import OpenGL.GL as gl
import OpenGL.GLU as glu
import sys

TAU = 2 * math.pi

ESTIMATE = False


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
    glu.gluCylinder(q, 2, 0, max(1, speed / 3), 20, 20)
    glu.gluDeleteQuadric(q)

    gl.glPopMatrix()


def draw_sphere(color, pos, radius=2):
    gl.glColor(*color)

    gl.glPushMatrix()

    x, y = pos
    gl.glTranslate(x, y, 3)

    q = glu.gluNewQuadric()
    glu.gluSphere(q, radius, 20, 20)
    glu.gluDeleteQuadric(q)

    gl.glPopMatrix()


elapsed = 0

def main(simulator):
    fig = glumpy.figure()
    world = fig.add_figure()

    @fig.event
    def on_init():
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glEnable(gl.GL_COLOR_MATERIAL)
        gl.glColorMaterial(gl.GL_FRONT_AND_BACK, gl.GL_AMBIENT_AND_DIFFUSE)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

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

        z = min(w, h) / 300.
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glTranslate(x + w / 2., y + h / 2., 10)
        gl.glScale(z, z, 1)

        a, b = simulator.agent.position
        gl.glTranslate(-a, -b, 0)

        gl.glLight(gl.GL_LIGHT0, gl.GL_POSITION, (0, 0, 100, 1))

        # draw lanes.
        gl.glLineWidth(2)
        gl.glColor(0, 0, 0)
        for lane in simulator.lanes:
            gl.glBegin(gl.GL_LINE_STRIP)
            for a, b in lane:
                gl.glVertex(a, b, 1)
            gl.glEnd()

        # draw cars.
        simulator.leader.draw(sys.modules[__name__], 1, 0, 0)
        simulator.agent.draw(sys.modules[__name__], 1, 1, 0)

        gl.glPopMatrix()

    @fig.event
    def on_idle(dt):
        global elapsed
        elapsed += dt
        while elapsed > simulator.dt:
            elapsed -= simulator.dt
            try:
                simulator.step()
            except StopIteration:
                sys.exit()
            fig.redraw()

    @fig.event
    def on_key_press(key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        elif key == glumpy.window.key.SPACE:
            global ESTIMATE
            ESTIMATE ^= ESTIMATE
        else:
            simulator.reset()
        fig.redraw()

    glumpy.show()
