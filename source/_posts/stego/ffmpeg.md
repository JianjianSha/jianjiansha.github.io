---
title: FFmpeg 源码分析
date: 2023-12-18 13:48:15
tags:
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