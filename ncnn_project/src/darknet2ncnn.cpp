/**
 * @File   : darknet2ncnn.cpp
 * @Author : damone (damonexw@gmail.com)
 * @Link   :
 * @Date   : 2018-10-25 10:23:01
 */

#include "darknet_tools.h"
#include <vector>

std::vector<split_data> split_info;

int get_ncnn_layer_count(layer l)
{
  LAYER_TYPE lt = l.type;
  int count = 1;
  if (lt == CONVOLUTIONAL)
  {
    count = 2;
    if (l.batch_normalize)
      count++;
  }
  else if (lt == DECONVOLUTIONAL)
  {
    count = 2;
    if (l.batch_normalize)
      count++;
  }
  else if (lt == LOCAL)
  {
    count = 2;
    if (l.batch_normalize)
      count++;
  }
  else if (lt == ACTIVE)
  {
    count = 1;
  }
  else if (lt == LOGXENT)
  {
    count = 1;
  }
  else if (lt == L2NORM)
  {
    count = 1;
  }
  else if (lt == RNN)
  {
    count = 1;
  }
  else if (lt == GRU)
  {
    count = 1;
  }
  else if (lt == LSTM)
  {
    count = 1;
  }
  else if (lt == CRNN)
  {
    count = 1;
  }
  else if (lt == CONNECTED)
  {
    count = 2;
    if (l.batch_normalize)
      count++;
  }
  else if (lt == CROP)
  {
    count = 1;
  }
  else if (lt == COST)
  {
    count = 0;
  }
  else if (lt == REGION)
  {
    count = 1;
  }
  else if (lt == YOLO)
  {
    count = 1;
  }
  else if (lt == DETECTION)
  {
    count = 1;
  }
  else if (lt == SOFTMAX)
  {
    count = 1;
  }
  else if (lt == NORMALIZATION)
  {
    count = 1;
  }
  else if (lt == BATCHNORM)
  {
    count = 1;
  }
  else if (lt == MAXPOOL)
  {
    count = 1;
  }
  else if (lt == REORG)
  {
    count = 1;
  }
  else if (lt == AVGPOOL)
  {
    count = 1;
  }
  else if (lt == ROUTE)
  {
    count = 1;
  }
  else if (lt == UPSAMPLE)
  {
    count = 1;
  }
  else if (lt == SHORTCUT)
  {
    count = 2;
  }
  else if (lt == DROPOUT)
  {
    count = 1;
  }
  else
  {
    fprintf(stderr, "Type not recognized: %d\n", l.type);
    count = 0;
  }
  return count;
}

void write_tag_data(FILE *bp, float *data, size_t size, unsigned int tag)
{
  fwrite(&tag, sizeof(unsigned int), 1, bp);
  fwrite(data, sizeof(float), size, bp);
}

void get_ncnn_batch_norm_form_layer(layer l, const char *layer_name_fmt,
                                    const char *input_fmt, const char *output_fmt,
                                    int layer_index, FILE *pp, FILE *bp)
{
  char format[256] = {0};
  sprintf(format, "%s %s 1 1  %s  %s", "%-16s", layer_name_fmt, input_fmt, output_fmt);
  fprintf(pp, format, "BatchNorm", layer_index, layer_index, layer_index);
  fprintf(pp, " 0=%d 1=0.00001\n", l.n);

  int n = l.n;
  if (l.type == CONNECTED)
  {
    n = l.outputs;
  }
  else if (l.type == BATCHNORM)
  {
    n = l.c;
  }

  // batch weights
  fwrite(l.scales, sizeof(float), n, bp);
  fwrite(l.rolling_mean, sizeof(float), n, bp);
  fwrite(l.rolling_variance, sizeof(float), n, bp);
  fwrite(l.biases, sizeof(float), n, bp);
}

void get_ncnn_layer_type_from_activation(ACTIVATION a, const char *layer_name_fmt,
                                         const char *input_fmt, const char *output_fmt,
                                         int layer_index, FILE *pp)
{
  char format[256] = {0};
  switch (a)
  {
  case LOGISTIC:
    sprintf(format, "%s %s 1 1 %s %s\n", "%-16s", layer_name_fmt, input_fmt, output_fmt);
    fprintf(pp, format, "Sigmoid", layer_index, layer_index, layer_index);
    break;
  case RELU:
    sprintf(format, "%s %s 1 1 %s %s 0=0.0\n", "%-16s", layer_name_fmt, input_fmt,
            output_fmt);
    fprintf(pp, format, "ReLU", layer_index, layer_index, layer_index);
    break;
  case ELU:
    sprintf(format, "%s %s 1 1 %s %s 0=1.0\n", "%-16s", layer_name_fmt, input_fmt,
            output_fmt);
    fprintf(pp, format, "ELU", layer_index, layer_index, layer_index);
    break;
  case SELU:
    sprintf(format, "%s %s 1 1 %s %s 0=1.6732\n", "%-16s", layer_name_fmt, input_fmt,
            output_fmt);
    fprintf(pp, format, "ELU", layer_index, layer_index, layer_index);
    break;
  case LEAKY:
    sprintf(format, "%s %s 1 1 %s %s 0=0.1\n", "%-16s", layer_name_fmt, input_fmt,
            output_fmt);
    fprintf(pp, format, "ReLU", layer_index, layer_index, layer_index);
    break;
  case TANH:
    sprintf(format, "%s %s 1 1 %s %s\n", "%-16s", layer_name_fmt, input_fmt, output_fmt);
    fprintf(pp, format, "TanH", layer_index, layer_index, layer_index);
    break;
  case RELIE:
  case RAMP:
  case LINEAR:
  case LOGGY:
  case PLSE:
  case STAIR:
  case HARDTAN:
  case LHTAN:
  {
    sprintf(format, "%s %s 1 1 %s %s %s\n", "%-16s", layer_name_fmt, input_fmt,
            output_fmt, "0=%d");
    fprintf(pp, format, "DarknetActivation", layer_index, layer_index,
            layer_index, a);
    break;
  }
  }
}

void parse_convolutional(network net, int layer_index, layer conv, FILE *pp,
                         FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  if (conv.groups == 1)
  {
    /**
     * conv
     */
    fprintf(pp, "%-16s conv_%d 1 1 %s conv_%d", "Convolution", layer_index,
            input_data.c_str(), layer_index);
    fprintf(pp, " 0=%d 1=%d 2=1 3=%d 4=%d 5=%d 6=%d\n", conv.n, conv.size,
            conv.stride, conv.pad, conv.batch_normalize == 0, conv.nweights);
  }
  else if (conv.groups > 1)
  {
    /**
     * dwconv
     */
    fprintf(pp, "%-16s conv_%d 1 1 %s conv_%d", "ConvolutionDepthWise",
            layer_index, input_data.c_str(), layer_index);
    fprintf(pp, " 0=%d 1=%d 2=1 3=%d 4=%d 5=%d 6=%d 7=%d\n", conv.n, conv.size,
            conv.stride, conv.pad, conv.batch_normalize == 0, conv.nweights,
            conv.groups);
  }

  // weights
  write_tag_data(bp, conv.weights, conv.nweights, 0x0);

  if (conv.batch_normalize)
  {
    /**
     * batch norm
     */
    get_ncnn_batch_norm_form_layer(conv, "conv_%d_batch_norm", " conv_%d",
                                   " conv_%d_batch_norm", layer_index, pp, bp);

    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(conv.activation, "conv_%d_activation",
                                        "conv_%d_batch_norm",
                                        "conv_%d_activation", layer_index, pp);
  }
  else
  {
    // biases
    fwrite(conv.biases, sizeof(float), conv.n, bp);
    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(conv.activation, "conv_%d_activation",
                                        "conv_%d", "conv_%d_activation",
                                        layer_index, pp);
  }
}

void parse_deconvolutional(network net, int layer_index, layer deconv, FILE *pp,
                           FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  if (deconv.groups == 1)
  {
    /**
     * deconv
     */
    fprintf(pp, "%-16s deconv_%d 1 1 %s deconv_%d", "Deconvolution",
            layer_index, input_data.c_str(), layer_index);
    fprintf(pp, " 0=%d 1=%d 2=1 3=%d 4=%d 5=%d 6=%d\n", deconv.n, deconv.size,
            deconv.stride, deconv.pad, deconv.batch_normalize == 0,
            deconv.nweights);
  }
  else if (deconv.groups > 1)
  {
    /**
     * deconv
     */
    fprintf(pp, "%-16s deconv_%d 1 1 %s deconv_%d", "DeconvolutionDepthWise",
            layer_index, input_data.c_str(), layer_index);
    fprintf(pp, " 0=%d 1=%d 2=1 3=%d 4=%d 5=%d 6=%d 7=%d\n", deconv.n,
            deconv.size, deconv.stride, deconv.pad, deconv.batch_normalize == 0,
            deconv.nweights, deconv.groups);
  }

  // weights
  write_tag_data(bp, deconv.weights, deconv.nweights, 0x0);

  if (deconv.batch_normalize)
  {
    /**
     * btach norm
     */
    get_ncnn_batch_norm_form_layer(deconv, "deconv_%d_batch_norm", " deconv_%d",
                                   "deconv_%d_batch_norm", layer_index, pp, bp);

    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(
        deconv.activation, "deconv_%d_activation", "deconv_%d_batch_norm",
        "deconv_%d_activation", layer_index, pp);
  }
  else
  {
    // biases
    fwrite(deconv.biases, sizeof(float), deconv.n, bp);

    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(
        deconv.activation, "deconv_%d_activation", "deconv_%d",
        "deconv_%d_activation", layer_index, pp);
  }
}

void parse_local(network net, int layer_index, layer local, FILE *pp,
                 FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  /**
   * local filters
   */
  int data_size =
      local.c * local.n * local.size * local.size * local.out_h * local.out_w;
  fprintf(pp, "%-16s local_%d 1 1 %s local_%d", "Not Impl", layer_index,
          input_data.c_str(), layer_index);
  fprintf(pp, " 0=%d 1=%d 2=1 3=%d 4=%d 5=%d\n", local.n, local.size,
          local.stride, local.pad, data_size);

  // weights
  write_tag_data(bp, local.weights, data_size, 0x0);
  // biases
  fwrite(local.biases, sizeof(float), local.outputs, bp);

  /**
   * activation
   */
  get_ncnn_layer_type_from_activation(local.activation, "local_%d_activation",
                                      "local_%d", "local_%d_activation",
                                      layer_index, pp);
}

void parse_active(network net, int layer_index, layer active, FILE *pp,
                  FILE *bp)
{
  if (layer_index <= 0)
    return;

  std::string input_format = get_layer_output_blob_format(net.layers[layer_index - 1]);
  if (split_info[layer_index - 1].splits.size() > 1)
  {
    input_format += split_info[layer_index - 1].get_split_info(layer_index);
  }

  get_ncnn_layer_type_from_activation(active.activation, "activation_%d",
                                      input_format.c_str(), "activation_%d",
                                      layer_index, pp);
}

void parse_logxent(network net, int layer_index, layer legxent, FILE *pp,
                   FILE *bp)
{
  if (layer_index <= 0)
    return;
  std::string input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);

  fprintf(pp, "%-16s logxent_%d 1 1 %s logxent_%d\n", "Sigmoid", layer_index,
          input_data.c_str(), layer_index);
}

void parse_l2norm(network net, int layer_index, layer l2norm, FILE *pp,
                  FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }
  fprintf(pp, "%-16s l2norm%d 1 1 %s l2norm%d 0=0 1=1 2=0.0 3=1 4=1\n",
          "Normalize", layer_index, input_data.c_str(), layer_index);

  // scale
  float scale = 1.0;
  write_tag_data(bp, &scale, 1, 0x0);
}

void parse_rnn(network net, int layer_index, layer rnn, FILE *pp, FILE *bp) {}

void parse_gru(network net, int layer_index, layer gru, FILE *pp, FILE *bp) {}

void parse_lstm(network net, int layer_index, layer lstm, FILE *pb, FILE *bp) {}

void parse_crnn(network net, int layer_index, layer crnn, FILE *pp, FILE *bp) {}

void parse_connected(network net, int layer_index, layer connected, FILE *pp,
                     FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  /**
   * connected
   */
  fprintf(pp, "%-16s connected_%d 1 1 %s connected_%d", "InnerProduct",
          layer_index, input_data.c_str(), layer_index);
  fprintf(pp, " 0=%d 1=%d 2=%d\n", connected.outputs,
          connected.batch_normalize == 0, connected.inputs * connected.outputs);

  // weights
  write_tag_data(bp, connected.weights, connected.outputs * connected.inputs,
                 0x0);

  if (connected.batch_normalize)
  {
    /**
     * btach norm
     */
    get_ncnn_batch_norm_form_layer(connected, "connected_%d_batch_norm",
                                   " connected_%d", " connected_%d_batch_norm",
                                   layer_index, pp, bp);

    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(
        connected.activation, "connected_%d_activation",
        "connected_%d_batch_norm", "connected_%d_activation", layer_index, pp);
  }
  else
  {
    // biases
    fwrite(connected.biases, sizeof(float), connected.outputs, bp);

    /**
     * activation
     */
    get_ncnn_layer_type_from_activation(
        connected.activation, "connected_%d_activation", "connected_%d",
        "connected_%d_activation", layer_index, pp);
  }
}

void parse_crop(network net, int layer_index, layer crop, FILE *pp, FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s crop_%d 1 1 %s crop_%d", "Crop", layer_index,
          input_data.c_str(), layer_index);
  int offset_h = (crop.h - crop.out_h) / 2;
  int offset_w = (crop.w - crop.out_w) / 2;
  fprintf(pp, " 0=%d 1=%d 2=0 3=%d 4=%d 5=-233\n", offset_w, offset_h,
          crop.out_w, crop.out_h);
}

void parse_cost(network net, int layer_index, layer cost, FILE *pp, FILE *bp) {}

void parse_region(network net, int layer_index, layer region, FILE *pp,
                  FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s region_%d 1 1 %s region_%d", "YoloDetectionOutput",
          layer_index, input_data.c_str(), layer_index);

  std::string biases = "";
  int biases_size = region.n * 2;
  for (int i = 0; i < biases_size; i++)
  {
    char value[32] = {0};
    sprintf(value, ",%f", region.biases[i]);
    biases += value;
  }

  fprintf(pp, " 0=%d 1=%d 2=0.25f 3=0.45f -23304=%d%s\n", region.classes,
          region.n, biases_size, biases.c_str());
}

void parse_yolo(network net, std::vector<int> yolos, FILE *pp, FILE *bp)
{
  std::string input_data = "";
  int input_count = 0;
  std::string biases = "";
  int last_index = 0;
  int classes = 0;
  int box_num = 0;
  int biases_size = 0;
  for (int i = 0; i < yolos.size(); i++)
  {
    last_index = yolos[i];
    layer yolo = net.layers[last_index];
    classes = yolo.classes;
    input_data += " " + get_layer_output_blob_name(split_info, last_index - 1, last_index);

    for (int n = 0; n < yolo.n; n++)
    {
      char value[32] = {0};
      int biases_index = 2 * yolo.mask[n];
      sprintf(value, ",%f,%f", yolo.biases[biases_index], yolo.biases[biases_index + 1]);
      biases += value;
      biases_size += 2;
      box_num++;
    }

    input_count++;
  }

  fprintf(pp, "%-16s yolo_%d %d 1 %s yolo_%d", "Yolov3Detection",
          last_index, input_count, input_data.c_str(), last_index);

  fprintf(pp, " 0=%d 1=%d 2=0 3=0.25f 4=0.45f 5=%d 6=%d -23307=%d%s\n",
          classes, box_num, net.w, net.h, biases_size, biases.c_str());
}

void parse_detection(network net, int layer_index, layer detection, FILE *pp,
                     FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s detection_%d 1 1 %s detection_%d", "Yolov1Detection",
          layer_index, input_data.c_str(), layer_index);
  fprintf(pp, " 0=%d 1=%d 2=%d 3=%d 4=%d 5=0.25f 6=0.45f\n", detection.side,
          detection.classes, detection.n, detection.sqrt, detection.softmax);
}

void parse_softmax(network net, int layer_index, layer softmax, FILE *pp,
                   FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s softmax_%d 1 1 %s softmax_%d 0=0\n", "Softmax",
          layer_index, input_data.c_str(), layer_index);
}

void parse_normalization(network net, int layer_index, layer norm, FILE *pp,
                         FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s normalization_%d 1 1 %s normalization_%d", "LRN",
          layer_index, input_data.c_str(), layer_index);
  fprintf(pp, " 0=1 1=%d 2=%f 3=%f 4=%f\n", norm.size, norm.alpha, norm.beta,
          norm.kappa);
}

void parse_batchnorm(network net, int layer_index, layer batchnorm, FILE *pp,
                     FILE *bp)
{
  if (layer_index <= 0)
    return;

  std::string input_format = get_layer_output_blob_format(net.layers[layer_index - 1]);
  if (split_info[layer_index - 1].splits.size() > 1)
  {
    input_format += split_info[layer_index - 1].get_split_info(layer_index);
  }

  get_ncnn_batch_norm_form_layer(batchnorm, "batch_norm_%d",
                                 input_format.c_str(), "batch_norm_%d",
                                 layer_index, pp, bp);
}

void parse_maxpool(network net, int layer_index, layer maxpool, FILE *pp,
                   FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s maxpool_%d 1 1 %s maxpool_%d", "Pooling", layer_index,
          input_data.c_str(), layer_index);
  int pad_left = maxpool.pad / 2;
  int pad_top = maxpool.pad / 2;
  int pad_right = maxpool.stride * maxpool.out_w + maxpool.size - maxpool.w - pad_left - 1;
  int pad_bottom = maxpool.stride * maxpool.out_h + maxpool.size - maxpool.h - pad_top - 1;
  fprintf(pp, " 0=0 1=%d 2=%d 3=%d 5=1 13=%d 14=%d 15=%d\n", maxpool.size,
          maxpool.stride, pad_left, pad_top, pad_right, pad_bottom);
}

void parse_reorg(network net, int layer_index, layer reorg, FILE *pp,
                 FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s reorg_%d 1 1 %s reorg_%d 0=%d\n", "Reorg", layer_index,
          input_data.c_str(), layer_index, reorg.stride);
}

void parse_avgpool(network net, int layer_index, layer avgpool, FILE *pp,
                   FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  fprintf(pp, "%-16s global_avg_pool_%d 1 1 %s global_avg_pool_%d 0=1 4=1\n",
          "Pooling", layer_index, input_data.c_str(), layer_index);
}

void parse_route(network net, int layer_index, layer route, FILE *pp,
                 FILE *bp)
{
  int i = 0;
  std::string input_data = "";
  for (; i < route.n; i++)
  {
    int index = route.input_layers[i];
    input_data += " " + get_layer_output_blob_name(split_info, index, layer_index);
  }

  fprintf(pp, "%-16s route_%d %d 1 %s route_%d 0=0\n", "Concat", layer_index,
          route.n, input_data.c_str(), layer_index);
}

void parse_upsample(network net, int layer_index, layer upsample, FILE *pp,
                    FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }
  fprintf(pp, "%-16s upsample_%d 1 1 %s upsample_%d", "Interp", layer_index,
          input_data.c_str(), layer_index);
  fprintf(pp, " 0=1 1=%d.f 2=%d.f\n", upsample.stride, upsample.stride);
}

void parse_shortcut(network net, int layer_index, layer shortcut, FILE *pp,
                    FILE *bp)
{
  std::string input_data = "";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, shortcut.index, layer_index);
    input_data += " " + get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }

  // commit, test, some network not work
  if (shortcut.c = shortcut.out_c && shortcut.h == shortcut.out_h &&
                   shortcut.w == shortcut.out_w) {
    fprintf(pp, "%-16s shortcut_%d 2 1 %s shortcut_%d 0=1", "Eltwise",
            layer_index, input_data.c_str(), layer_index);
    if (1.f != shortcut.alpha || 1.f != shortcut.beta) {
      fprintf(pp, " -23301=2,%f,%f", shortcut.beta, shortcut.alpha);
    }
    fprintf(pp, "\n");
  } else
  {
    fprintf(pp, "%-16s shortcut_%d 2 1 %s shortcut_%d 0=%f 1=%f\n",
            "DarknetShortcut", layer_index, input_data.c_str(), layer_index,
            shortcut.alpha, shortcut.beta);
  }

  /**
   * activation
   */
  get_ncnn_layer_type_from_activation(
      shortcut.activation, "shortcut_%d_activation", "shortcut_%d",
      "shortcut_%d_activation", layer_index, pp);
}

void parse_dropout(network net, int layer_index, layer dropout, FILE *pp,
                   FILE *bp)
{
  std::string input_data = "data";
  if (layer_index > 0)
  {
    input_data = get_layer_output_blob_name(split_info, layer_index - 1, layer_index);
  }
  fprintf(pp, "%-16s dropout_%d 1 1 %s dropout_%d\n", "Dropout", layer_index,
          input_data.c_str(), layer_index);
  // if set, should be 1.f
  // fprintf(pp, " 0=%f\n", dropout.scale);
}

void parse_network(char *cfgfile, char *weightfile, char *ncnn_prototxt,
                   char *ncnn_modelbin)
{
  network *net_p = load_network(cfgfile, weightfile, 0);
  network net = *net_p;

  FILE *pp = fopen(ncnn_prototxt, "wb");
  FILE *bp = fopen(ncnn_modelbin, "wb");

  split_info.clear();
  get_layers_split_info(*net_p, split_info);
  int blob_count = 1, i = 0;
  int ncnn_layer_count = 1;
  for (i = 0; i < net.n; ++i)
  {
    if (net.layers[i].type == YOLO && i != net.n - 1)
      continue;

    blob_count += get_ncnn_layer_count(net.layers[i]);
    ncnn_layer_count += get_ncnn_layer_count(net.layers[i]);

    if (split_info[i].splits.size() > 1)
    {
      ncnn_layer_count++;
      blob_count += split_info[i].splits.size();
    }
  }

  fprintf(pp, "7767517\n");
  fprintf(pp, "%d %d\n", ncnn_layer_count, blob_count);
  fprintf(pp, "%-16s data 0 1 data 0=%d 1=%d 2=%d\n", "Input", net.w, net.h, net.c);

  std::vector<int> yolos;
  for (i = 0; i < net.n; ++i)
  {
    layer l = net.layers[i];
    LAYER_TYPE lt = l.type;
    if (lt == CONVOLUTIONAL)
    {
      parse_convolutional(net, i, l, pp, bp);
    }
    else if (lt == DECONVOLUTIONAL)
    {
      parse_deconvolutional(net, i, l, pp, bp);
    }
    else if (lt == LOCAL)
    {
      parse_local(net, i, l, pp, bp);
    }
    else if (lt == ACTIVE)
    {
      parse_active(net, i, l, pp, bp);
    }
    else if (lt == LOGXENT)
    {
      parse_logxent(net, i, l, pp, bp);
    }
    else if (lt == L2NORM)
    {
      parse_l2norm(net, i, l, pp, bp);
    }
    else if (lt == RNN)
    {
      parse_rnn(net, i, l, pp, bp);
    }
    else if (lt == GRU)
    {
      parse_gru(net, i, l, pp, bp);
    }
    else if (lt == LSTM)
    {
      parse_lstm(net, i, l, pp, bp);
    }
    else if (lt == CRNN)
    {
      parse_crnn(net, i, l, pp, bp);
    }
    else if (lt == CONNECTED)
    {
      parse_connected(net, i, l, pp, bp);
    }
    else if (lt == CROP)
    {
      parse_crop(net, i, l, pp, bp);
    }
    else if (lt == COST)
    {
      parse_cost(net, i, l, pp, bp);
    }
    else if (lt == REGION)
    {
      parse_region(net, i, l, pp, bp);
    }
    else if (lt == YOLO)
    {
      yolos.push_back(i);
    }
    else if (lt == DETECTION)
    {
      parse_detection(net, i, l, pp, bp);
    }
    else if (lt == SOFTMAX)
    {
      parse_softmax(net, i, l, pp, bp);
    }
    else if (lt == NORMALIZATION)
    {
      parse_normalization(net, i, l, pp, bp);
    }
    else if (lt == BATCHNORM)
    {
      parse_batchnorm(net, i, l, pp, bp);
    }
    else if (lt == MAXPOOL)
    {
      parse_maxpool(net, i, l, pp, bp);
    }
    else if (lt == REORG)
    {
      parse_reorg(net, i, l, pp, bp);
    }
    else if (lt == AVGPOOL)
    {
      parse_avgpool(net, i, l, pp, bp);
    }
    else if (lt == ROUTE)
    {
      parse_route(net, i, l, pp, bp);
    }
    else if (lt == UPSAMPLE)
    {
      parse_upsample(net, i, l, pp, bp);
    }
    else if (lt == SHORTCUT)
    {
      parse_shortcut(net, i, l, pp, bp);
    }
    else if (lt == DROPOUT)
    {
      parse_dropout(net, i, l, pp, bp);
    }
    else
    {
      fprintf(stderr, "Type not recognized: %d\n", l.type);
      break;
    }

    /**
     * if splits > 1, to split
     */
    if (split_info[i].splits.size() > 1)
    {
      split_data &split = split_info[i];
      std::string input_name = split.output_blob_name;
      std::string outputs = "";
      splits_type::iterator iter = split.splits.begin();
      for (; iter != split.splits.end(); iter++)
      {
        outputs += " " + split.output_blob_name + iter->second;
      }

      int output_count = split.splits.size();
      fprintf(pp, "%-16s %s_split 1 %d %s %s\n", "Split", input_name.c_str(), output_count, input_name.c_str(), outputs.c_str());
    }
  }

  if (yolos.size() > 0)
  {
 
    parse_yolo(net, yolos, pp, bp);
  }

  fclose(bp);
  fclose(pp);
}

int main(int argc, char *argv[])
{
  if (argc < 5)
  {
    printf("use : %s darknet.cfg darknet.weights ncnn.param ncnn.bin", argv[0]);
    return -1;
  }
  parse_network(argv[1], argv[2], argv[3], argv[4]);
  return 0;
}