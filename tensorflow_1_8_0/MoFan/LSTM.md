# Static RNN, Dynamic RNN
* 差異:
  Static RNN 快訓練預測都比較快(也就是Rolling number)
  Dynamic RNN 可以接受不同的data dimension
  

> 从直觉上很容易觉察到，动态RNN在功能上有很多优势（允许样本长度不同），然而天下没有免费的午餐，静态RNN肯定有动态RNN不具备的优势，否则静态RNN早就该废弃了。静态RNN会把RNN展开成多层，这样似乎相当于用空间换时间。动态RNN使用while循环，这样相当于用时间换空间。静态RNN在运行速度上会比动态RNN快。

  