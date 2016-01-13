#!/usr/bin/env python

import sys
import xml.parsers.expat

TASKS = ('Speedometer', 'LeadCar')

class Fixation(list):
    def __init__(self, attrs):
        self.attrs = attrs

    def to_str(self, frame):
        s = '%s %s %s %s %s' % (frame,
            self.attrs['horiz'], self.attrs['vert'],
            self.attrs['screen_x'], self.attrs['screen_y'])
        for x in self:
            if x.name in TASKS:
                s += ' ' + str(x)
                break
        return s

    def is_relevant(self):
        return set(TASKS) & set(x.name for x in self)

    def set(self, key, value):
        setattr(self[-1], key, value.strip().replace(' ', ''))


class FixatedItem:
    FIELDS = ('id', 'percent', 'x', 'y')

    def __init__(self, attrs):
        self.percent = attrs['percent']
        self.name = None
        self.x = None
        self.y = None

    @property
    def id(self):
        return TASKS.index(self.name)

    def __str__(self):
        return ' '.join(str(getattr(self, k)) for k in self.FIELDS)


class Parser:
    def __init__(self):
        self.names = []
        self.fixation = None
        self.frame = 0
        self.text = u''

    @property
    def text(self):
        return self.texts[-1]

    def start_element(self, name, attrs):
        self.names.append(name)
        if name == 'frame':
            self.frame = int(attrs['number'])
        if name == 'fixated':
            self.fixation = Fixation(attrs)
        if name == 'item' and self.fixation is not None:
            self.fixation.append(FixatedItem(attrs))

    def end_element(self, name):
        #print '#', u'/'.join(self.names)
        assert name == self.names.pop()

        if name == 'fixated':
            if self.fixation.is_relevant():
                print self.fixation.to_str(self.frame)
            self.fixation = None
            self.text = u''
            return

        if 'fixated' not in self.names:
            self.text = u''
            return

        if 'item' in self.names:
            if name in ('x', 'y') and self.names[-1] == 'screen_coords':
                self.fixation.set(name, self.text)
            if name in ('name'):
                self.fixation.set(name, self.text)

        self.text = u''

    def char_data(self, data):
        self.text += data

    def parse(self, file):
        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = self.start_element
        p.EndElementHandler = self.end_element
        p.CharacterDataHandler = self.char_data
        p.ParseFile(file)


if __name__ == '__main__':
    Parser().parse(sys.stdin)
