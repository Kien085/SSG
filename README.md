#SGAE - Generating Scene Graphs from Captions
1. First download the [Stanford CoreNLP 3.6.0](http://stanfordnlp.github.io/CoreNLP/index.html) code and models. To do this, run ./get_stanford_models.sh inside of the coco-caption folder
2. Then run create_coco_sg.py. This will create a file name sg.json inside of /coco-caption/pycocoevalcap/spice. Rename this file to spice_sg_train.json and move it into the /data folder
3. Next go into create_coco_sg.py and change the coco_use to coco_val. Run it, and this time save the file as spice_sg_val.json and move it into the /data folder
4. Use process_spice_sg.py to create coco_pred_sg, which will populate /data/coco_spice_sg2, which is where the scene graphs are, as well as data/spice_sg_dict2.npz and data/spice_sg_dict_raw2.npz, which are the dictionaries for those scene graphs
5. If you want to show the scene graph, you can run the following code, replacing the id with the id of the image that you want to see the scene graph of.
```
python show_sg.py --mode sen --id 391895
```

To generate the visualization, go into the vis folder, and call

```
python test.py --mode sen --id 391895
```