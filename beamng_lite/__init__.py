"""BeamNG.Drive Lite — мини soft-body автосимулятор на PyOpenGL."""
try:                      # настройки PyOpenGL до импорта OpenGL.GL
    import OpenGL
    OpenGL.ERROR_CHECKING = False     # заметно быстрее в цикле отрисовки
    OpenGL.ERROR_LOGGING = False
except ImportError:       # headless-режим работает и без PyOpenGL
    pass
