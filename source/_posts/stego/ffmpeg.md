---
title: FFmpeg 源码分析
date: 2023-12-18 13:48:15
tags: ffmpeg
---

**# 1 创建编码器**

```c++
const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
```

此时 `libx264.c` 文件中的变量

```c++
FFCodec ff_libx264_encoder = { 
    .p.name = "libx264",
    CODEC_LONG_NAME("libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 30"),
    .p.type = AVMEDIA_TYPE_VIDEO,
    .p.id = AV_CODEC_ID_H264,
    .p.capabilities = ...,
    .p.priv_class = &x264_class,
    .p.wrapper_name = "libx264",
    .priv_data_size = sizeof(X264Context),
    .init = X264_init,
    ...
};
```

实际是返回的就是 `ff_libx264_encoder` 对象的第一个字段 `p` 的地址（实际上也是 `ff_libx264_enoder` 的地址）

这是因为 `FFCodec` 结构体中第一个字段类型就是 `AVCodec`，而方法

```c++
const AVCodec *av_codec_iterate(void **opaque) {
    uintptr_t i = (uintptr_t)*opaque;   // i 初始值为 0
    const FFCodec *c = codec_list[i];   // 遍历 codec_list 中的对象
    if (c) {
        *opaque = (void*)(i+1);         // codec_list 的 index 前进一位
        return &c->p;                   // 返回 AVCodec 类型的地址
    }
}
```

**# 2 创建编码器上下文对象**


```c++
AVCodecContext* ctx = avcodec_alloc_context3(codec);
```

源码是

```c++
AVCodecContext *avcodec_alloc_context3(const AVCodec *codec) {
    AVCodecContext *avctx = av_malloc(sizeof(AVCodecContext));  // 分配对象内存
    if (!avctx) { return NULL; }
    if (init_context_defaults(avctx, codec) < 0) {
        av_free(avctx);
        return NULL;
    }
    return avctx;
}
```

这里最重要的就是初始化函数，初始化 AVCodecContext 到底设置了什么，

```c++
// options.c 
static int init_context_defaults(AVCodecContext *s, const AVCodec *codec) {
    // 直接类型强转即可，因为指向同一个地址
    const FFCodec *const codec2 = ffcodec(codec);
    int flags = 0;
    memset(s, 0, sizeof(AVCodecContext));

    s->av_class = &av_codec_context_class;  // 见下方代码

    // AVMEDIA_TYPE_VIDEO
    s->codec_type = codec ? codec->type : AVMEDIA_TYPE_UNKNOWN;
    if (codec) {
        s->codec = codec;
        s->codec_id = codec->id;
    }

    ...
    // flags = AV_OPT_FLAGS_VIDEO_PARAM;
    av_opt_set_defaults2(s, flags, flags);
    ... // 后面在继续分析，这里先暂停，跳转去分析 av_opt_set_defaults2 这个函数
}

static const AVClass av_codec_context_class = {
    .class_name = "AVCodecContext",
    .item_name = context_to_name,
    .option = avcodec_options,
    ...
}
```

`av_opt_set_defaults2` 顾名思义就是设置一些默认值，我们看下设置了什么，

```c++
const AVOption *av_opt_next(const void *obj, const AVOption *last) {
    const AVClass *class;
    if (!obj) return { NULL; }
    // 这里 obj 就是 AVCodecContext 指针

    // 类型强转，那肯定是两个指针指向同一个地址，见下方 AVCodecContext 结构体定义
    class = *(const AVClass**)obj;
    if (!last && class && class->option && class->option[0].name)
        return class->option;
    if (last && last[1].name)
        return ++last;
    return NULL;
}
```

上面代码中 `class->option` 位于 `options_table.h` 中，包含了所有的选项设置，这是一个数组，
所以上面代码中后续返回的是这个数组中下一个选项，即这是一个遍历数组中选项的函数。

```c++
typedef struct AVCodecContext {
    const AVClass *av_class;
    ...
}
```

```c++
void av_opt_set_defaults2(void *s, int mask, int flags) {
    const AVOption *opt = NULL;
    // 传入参数：AVCodecContext 指针，上一个 AVOption 指针（初始时上一个为 NULL）
    while ((opt = av_opt_next(s, opt))) { // 遍历所有选项
        
    }
}
```

所有选项见下方定义，创建一个编码器时，这些选项的值决定了编码过程，所以弄清楚这些编码选项非常重要。

```c++
// options_table.h
static const AVOption avcodec_options[] = {
{"b", "set bitrate (in bits/s)", OFFSET(bit_rate), AV_OPT_TYPE_INT64, {.i64 = AV_CODEC_DEFAULT_BITRATE }, 0, INT64_MAX, A|V|E},
...
}
```

关于选项的说明可以参见[官方文档](https://ffmpeg.org/ffmpeg-codecs.html#Codec-Options) ，不完全一样，因为官方文档给的是 `ffmpeg` 这个命令的选项，但是可以参考一下。

例如，选项 `b` 表示设置比特率，

```sh
ffmpeg -b:v 1000k   # 设置视频流比特率
ffmpeg -b:a 128k    # 设置音频流比特率
```

编码相关参数设置，参考 [Encode/H.264](https://trac.ffmpeg.org/wiki/Encode/H.264)

回到 avcodec_options 中的选项定义中，第一个选项设置比特率，其中

```c++
OFFSET(bit_rate) -> offsetof(AVCodecContext, bit_rate)
```
计算的是 `bit_rate` 这个成员变量在 `AVCodecContext` 中的偏移量，结果是 `56`，即偏移了 56 个 byte。

至此，我们已经知道 `av_opt_set_defaults2` 就是将选项的默认值设置到 `AVCodecContext` 对应的各个变量上，接着我们再看 `init_context_defaults` 这个方法还做了哪些事情，

```c++
// options.c 
static int init_context_defaults(AVCodecContext *s, const AVCodec *codec) {
    const FFCodec *const codec2 = ffcodec(codec);   // 类型强转，因为指向同一个地址
    ...

    s->time_base            = (AVRational){0, 1};   // 时间基，默认值为 0
    s->framerate            = (AVRational){0, 1};   // 帧率，默认值为 0
    ...
    s->pix_fmt              = AV_PIX_FMT_NONE;
    ...

    if (codec && codec2->priv_data_size) {
        // 分配内存空间并全 0 初始化这个内存空间，以 x264 为例，这个空间对应的是 X264Context
        // 见 ff_libx264_encoder 定义
        s->priv_data = av_mallocz(codec2->priv_data_size);
        if (!s->priv_data) return AVERROR(ENOMEM);
        if (codec->priv_class) {
            // 根据优先级，先指针成员选择，然后类型强转，所以是将 AVCodecContext 的 priv_data
            // 从 void* （实际上是 X264Context*）转为 AVClass*，而 AVClass 是 X264Context
            // 第一个成员变量的类型，所以指向同一个地址，可以强转，故其实是设置 X264Context
            // 的第一个成员变量 class 为 x264_class
            *(const AVClass**)s->priv_data = codec->priv_class;
            // 设置下方的 `options` 选项（即 preset tune profile 等）的默认值到 X264Context 的变量上
            av_opt_set_defaults(s->priv_data);
        }
    }
    if (codec && codec2->defaults) {
        int ret;
        const FFCodecDefault *d = codec2->defaults; // 设置各个编码器自带的一些默认值
                                                    // 见下方的 `x264_defaults` 变量定义
        while (d->key) {
            ret = av_opt_set(s, d->key, d->value, 0);// 设置到 AVCodecContext 中
            av_assert0(ret >= 0);
            d++;
        }
    }
    return 0;
}
```

以 libx264 为例，上述代码中 `codec->priv_class` 就是 `ff_libx264_encoder` 中的 `&x264_class`，此变量定义如下，

```c++
static const AVClass x264_class = {
    .class_name = "libx264",
    .item_name = av_default_item_name,
    .option = options,
    .version = LIBAVUTIL_VERSION_INT,
};
```

其中 `options` 这个变量定义在 `libx264.c` 文件中，此变量是一个数组，包含了一些 x264 自有的一些选项设置，与上面 `avcodec_options` 的格式类似，

```c++
static const AVOption options[] = {
    { "preset", "Set the encoding preset", OFFSET(preset), AV_OPT_TYPE_STRING, { .str = "medium" }, 0, 0, VE},
    { "tune", "Tune the encoding params", OFFSET(tune), AV_OPT_TYPE_STRING, {0}, 0, 0, VE},
    { "profile", "Set profile restrictions", OFFSET(profile_opt), AV_OPT_TYPE_STRING, {0}, 0, 0, VE},
    ...
};
```

但是这里的选项不是设置到 `AVCodecContext` 中的变量，而是设置到 `X264Context` 的变量上，所以，`X264Context` 的 `preset` 变量默认值为 `"medium"`，而 `tune` 和 `profile` 两个变量的默认值为 `""`。

```c++
// libx264.c

// 这些选项默认值将设置到 AVCodecContext 中的相应变量上去。
static const FFCodecDefault x264_defaults[] = {
    { "b",          "0" },  // "b" 是比特率，见 avcodec_options 的中的选项定义
    { "bf",         "-1" },
    { "flags2",     "0" },
    ...
    { "qmin",       "-1" },
    { "qmax",       "-1" },
    { "qblur",      "-1" },
    { "qcomp",      "-1" },
    ...
}
```
所以这里将编码器的比特率又重新设置为 `0` 。


至此，完成了 `AVCodecContext` 的对象创建以及成员变量的默认赋值。

## 2. 关联上下文

```c++
// avcodec.c
int avcodec_open2(AVCodecContext *avctx, const AVCodec *codec, AVDictionary **options) {
    AVCodecInternal *avci;
    const FFCodec *codec2;

    codec2 = ffcodec(codec);
    avctx->codec_type = codec->type;
    avctx->codec_id = codec->id;
    avctx->codec = codec;

    avci = av_codec_is_decoder(codec) ?
        ff_decode_internal_alloc() :
        ff_encode_internal_alloc();

    avctx->internal = avci;
    avci->buffer_frame = av_frame_alloc();      // 分配内部 frame 缓存
    avci->buffer_pkt = av_packet_alloc();       // 分配内部 packet 缓存

    // 将 width 和 height 分别设置到 coded_width 和 coded_height
    ret = ff_set_dimensions(avctx, avctx->width, avctx->height);

    if (av_codec_is_encoder(avctx->codec))
        ret = ff_encode_preinit(avctx); // 这里面又做了很多 validating 和初始化工作
    else
        ret = ff_decode_preinit(avctx);

    ...

    if (!(avctx->active_thread_type & FF_THREAD_FRAME) ||
        avci->frame_thread_encoder) {// frame_thread_encoder 有效，见 ff_frame_thread_encoder_init 函数
        if (codec2->init) {
            lock_avcodec(codec2);
            ret = codec2->init(avctx);  // init 方法为 X264_init，参见 libx264.c 文件
            unlock_avcodec(codec2);
        }
    }
    ...
}
```

上述代码设置了编码器的一些参数，包括分配内部的 frame 和 packet 缓存空间，设置帧的宽高，以及多线程编码上下文对象等。与 x264 相关的初始化工作则在 `X264_init` 中完成，

```c++
static int X264_init(AVCodecContext *avctx) {
    X264Context *x4 = avctx->priv_data;
    x264_param_default(&x264->params);

    x4->params.b_deblocking_filter = avctx->flags & AV_CODEC_FLAG_LOOP_FILTER;

    if (x4->preset || x4->tune) {
        if (x264_param_default_preset(&x4->params, x4->preset, x4->tune) < 0) {
            ...
        }
    }
}
```

以上代码中，`x264_param_default` 用于设置一些默认参数值，具体可参见 x264 源码中的 `base.c` 文件。[x264 git 仓库地址](https://code.videolan.org/videolan/x264.git)

`x264_param_default_preset` 则根据 `preset` 和 `tune` 选项值来设置参数。

**# 根据 preset 设置参数**

代码位于 x264 源码中的 `base.c` 文件的 `param_apply_preset` 函数。

**# 根据 tune 设置参数**

代码位于 x264 源码中的 `base.c` 文件的 `param_apply_tune` 函数。

**# 根据 profile 设置参数**

```c++
// libx264.c  X264_init 函数
x4->profile = x4->profile_opt;  // 这里 profile_opt 默认值为 ""，见本文件 options 变量
if (!x4->profile) {
    switch (avctx->profile) {   // 见 options_table.h 中的 avcodec_options 变量
        ...
    }
}

if (x4->profile) {
    // 此函数位于 x264 源码的 base.c 文件
    if (x264_param_apply_profile(&x4->params, x4->profile) < 0) { ... }
}
```


## 3. 编码

```c++
// encode.c

int avcodec_send_frame(AVCodecContext* avctx, const AVFrame* frame) {
    AVCodecInternal *avci = avctx->internal;
    ...
    
    if (avci->buffer_frame->buf[0]) // 输入在当前状态不被接受，try again
        return AVERROR(EAGAIN);
    if (!frame) {
        avci->draining = 1;     // 对应最后的 null frame 冲刷编码器
    } else {
        ret = encode_send_frame_internal(avctx, frame);
        if (ret < 0) return ret;
    }
    
    if (!avci->buffer_pkt->data && !avci->buffer_pkt->side_data) {  // 内部缓存的编码后包数据为 NULL，则调用下方函数进行编码
        ret = encode_receive_packet_internal(avctx, avci->buffer_pkt);
        if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF)
            return ret;     // 编码出错
    }
    ...
}
```

此方法对编码器上下文进行了判断，例如是否成功 open，是否是编码器，是否进入 draining 模式等。对于一个进来的正常的帧，然后内部调用 `encode_send_frame_internal` 方法，但是这个方法除了将待编码的帧数据拷贝到内部缓存中，其他也没做什么事情。

```c++
static int encode_send_frame_internal(AVCodecContext *avctx, const AVFrame* src) {
    AVCodecInternal *avci = avctx->internal;
    AVFrame *dst = avci->buffer_frame;
    int ret;
    ret = av_frame_ref(dst, src);   // 将 src 指向的数据拷贝到 dst 指向的对象
    if (ret < 0) return ret;

    if (avctx->codec->type == AVMEDIA_TYPE_VIDEO) {
        ret = encode_generate_icc_profile(avctx, dst);
        if (ret < 0) return ret;
    }
    if (!(avctx->flags & AV_CODEC_FLAG_FRAME_DURATION)) // 由于flags默认值为0（参见默认选择设置 avcodec_options）
        dst->duration = 0;
}
```

记住拷贝的帧数据存储在 `avci->buffer_frame` 中。然后就是对拷贝到缓存的帧数据进行编码，方法定义如下，

```c++
// encode.c
// 编码，编码后数据存储到 AVPacket 中
static int encode_receive_packet_internal(AVCodecContext* avctx, AVPacket* avpkt) {
    ...
    int ret = encode_simple_receive_packet(avctx, avpkt);
}

static int encode_simple_receive_packet(AVCodecContext *avctx, AVPacket *avpkt) {
    int ret;

    while (!avpkt->data && !avpkt->side_data) { // data 和 side_data 均为 NULL，则继续循环
        ret = encode_simple_internal(avctx, avpkt);
        if (ret < 0) return ret;
    }
    return 0;
}

static int encode_simple_internal(AVCodecContext *avctx, AVPacket *avpkt) {
    AVCodecInternal *avci = avctx->internal;
    AVFrame *frame = avci->in_frame;
    const FFCodec *const codec = ffcodec(avctx->codec);
    int got_packet;
    int ret;
    ...
    if (!frame->buf[0] && !avci->draining) {    // frame 没有数据，并且没有进入 draining 状态
                                                // 记住送入的帧数据拷贝到 avci->buffer_frame
        av_frame_unref(frame);  // 对 avci->in_frame 解旧引用，准备引用新的帧数据
        ret = ff_encode_get_frame(avctx, frame);// 将 avci->buffer_frame 关联到 avci->in_frame 上
        if (ret < 0 && ret != AVERROR_EOF)
            return ret;
    }
    ...
    got_packet = 0;
    if (CONFIG_FRAME_THREAD_ENCODER && avci->frame_thread_encoder)
        ret = ff_thread_video_encoder_frame(avctx, avpkt, frame, &got_packet);//线程编码实现
    else
        ret = ff_encode_encode_cb(avctx, avpkt, frame, &got_packet);
}
```
