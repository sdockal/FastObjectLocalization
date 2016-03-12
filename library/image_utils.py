import urllib2, os, tempfile
import sys
import cv2
from socket import error as SocketError

def image_from_url(url):
  """
  Read an image from a URL. Returns a numpy array with the pixel data.
  We write the image to a temporary file then read it back. Kinda gross.
  """
  try:
    f = urllib2.urlopen(url)
    _, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
      ff.write(f.read())
    img = cv2.imread(fname)
    os.remove(fname)
    return img
  except urllib2.URLError as e:
    print 'URL Error: ', e.reason, url
  except urllib2.HTTPError as e:
    print 'HTTP Error: ', e.code, url
  except ValueError as e:
    print 'Value Error: ', url
  except SocketError as e:
    print 'Socket Error: ', url
  except httplib.BadStatusLine as e:
    print 'Bad StatusLine Error: ', url
