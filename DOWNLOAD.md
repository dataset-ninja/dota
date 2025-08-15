Dataset **DOTA** can be downloaded in [Supervisely format](https://developer.supervisely.com/api-references/supervisely-annotation-json-format):

 [Download](https://assets.supervisely.com/remote/eyJsaW5rIjogInMzOi8vc3VwZXJ2aXNlbHktZGF0YXNldHMvMjU0MV9ET1RBL2RvdGEtRGF0YXNldE5pbmphLnRhciIsICJzaWciOiAieWd3bzJiNTA0eUdFd2R0ZHZKdytHK2VrUys4aU9odTMwL0Y1bEluYTZ0dz0ifQ==?response-content-disposition=attachment%3B%20filename%3D%22dota-DatasetNinja.tar%22)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='DOTA', dst_dir='~/dataset-ninja/')
```
Make sure not to overlook the [python code example](https://developer.supervisely.com/getting-started/python-sdk-tutorials/iterate-over-a-local-project) available on the Supervisely Developer Portal. It will give you a clear idea of how to effortlessly work with the downloaded dataset.

The data in original format can be [downloaded here](https://captain-whu.github.io/DOTA/dataset.html).