import glumpy
import math
import OpenGL.GL as gl
import OpenGL.GLU as glu

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


def switch_alphas():
    global ESTIMATE_ALPHA, REALITY_ALPHA
    ESTIMATE_ALPHA, REALITY_ALPHA = REALITY_ALPHA, ESTIMATE_ALPHA


def main(opts, tracks, leader, agent):
    DT = 1. / opts.sample_rate
    elapsed = 0
    frame = 0

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
        for track in tracks:
            gl.glBegin(gl.GL_LINE_STRIP)
            for a, b in track:
                gl.glVertex(a, b, 1)
            gl.glEnd()

        # draw cars.
        agent.draw(gfx, 1, 1, 0, gfx.REALITY_ALPHA)
        leader.draw(gfx, 1, 0, 0, gfx.REALITY_ALPHA)

        gl.glPopMatrix()

    @fig.event
    def on_idle(dt):
        global elapsed, frame

        #elapsed += DT  # -- run the simulation at top speed
        elapsed += dt  # -- run the simulation at real-time

        while elapsed > DT:
            elapsed -= DT
            frame += 1

            agent.move(DT)
            leader.move(DT)

            if not frame % int(opts.sample_rate / 3):
                agent.update(leader)

            print numpy.linalg.norm(leader.target - agent.position),
            print agent.speed - opts.target_speed,
            for m in agent.modules:
                print m.salience,
            print

        fig.redraw()

    @fig.event
    def on_key_press(key, modifiers):
        if key == glumpy.window.key.ESCAPE:
            sys.exit()
        elif key == glumpy.window.key.SPACE:
            gfx.switch_alphas()
        else:
            leader.reset()
            agent.reset(leader)
        fig.redraw()

    glumpy.show()
