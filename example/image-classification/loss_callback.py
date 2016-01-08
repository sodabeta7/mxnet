import find_mxnet
import mxnet as mx
import numpy as np
import logging
from itertools import izip
from sklearn.metrics import log_loss

def get_loss_batch_callback(model, period):
  wd = model.kwargs['wd']  
  arg_names = model.symbol.list_arguments()
  weight_names = [name for name in arg_names if name.endswith('weight')]
  metric = []  
  def _callback(param):
    execm = param.executor_manager
    labels = param.batch_label
    preds = execm.cpu_output_arrays
    loss = 0
    for label, pred in izip(labels, preds):
      label = label.asnumpy()
      pred = pred.asnumpy()
      loss += log_loss(label, pred)
    weight = dict(zip(execm.param_names, execm.param_arrays))
    oloss = loss
    for name in weight_names:      
      w_block = map(lambda x:x.asnumpy(), weight[name])
      for w in w_block:
        assert np.allclose(w, w_block[0])
      loss += wd*np.sum(w_block[0]**2)/2.0            
    logging.info('Iter[%d] Batch[%d] Train-%s=%f wd %f sm %f',
                  param.epoch, param.nbatch, 'loss', loss, loss-oloss, oloss)

  return _callback

