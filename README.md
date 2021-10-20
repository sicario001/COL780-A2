# Template Tracking
## How to Run
#### Part 1
- Code is in `blockBasedTracker.py`
- The hyper parameters will have to be selected (line 10) and directory of images too (line 97).
- `python blockBasedTracker.py`
#### Part 2 and 3
##### Affine
- Code is in `templateTrackingAffine.py`
- Set hyper parameters and directory as before.
- `python templateTrackingAffine.py`
##### Translation
- Code is in `template-tracking-translation.py`
- Set hyper parameters and directory as before.
- `python template-tracking-translation.py`
#### Part 4
- Code is in `live-demo.py`
```
export FLASK_APP=live-demo
flask run
```

- Use the set scale field to resize the template image.
