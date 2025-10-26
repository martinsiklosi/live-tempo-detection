# TODO

## Add tempo templates

Want to be able to from beat annotation create and plot a tempo curve so you can follow it.

## Improve no tempo detection

Perhaps with a volume threshhold.

## Imrpove startup time

See if lazy imports are possible
Import matplotlib and librosa only when needed?
Can audio recording and estimation be started while matplotlib is importing?
Librosa is imported in data models but is only meeded for reading audio from file
Scipy stft is faster to load than librosa