"""Exception to be thrown by the hol_light API in case of errors.

Note that the naming of this file is chosen so that it can match the
Google internal version for compatibility.
"""


class StatusNotOk(Exception):  # pylint: disable=g-bad-exception-name

  def __init__(self, msg):
    self.message = msg
    Exception.__init__(self, str(self))

  def __str__(self):
    return self.message
