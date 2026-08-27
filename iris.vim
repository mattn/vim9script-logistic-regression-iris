vim9script
def Softmax(w: list<float>, x: list<float>): float
  var v = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]
  return 1.0 / (1.0 + exp(0.0 - v))
enddef

def LogisticRegression(X: list<list<float>>, y: list<float>, rate: float, ntrains: number): list<float>
  var w0 = (rand() / 4294967295.0 - 0.5) * 2.0
  var w1 = (rand() / 4294967295.0 - 0.5) * 2.0
  var w2 = (rand() / 4294967295.0 - 0.5) * 2.0
  var w3 = (rand() / 4294967295.0 - 0.5) * 2.0
  for n in range(ntrains)
    for i in range(len(X))
      var [x0, x1, x2, x3] = X[i]
      var v = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3
      var pred = 1.0 / (1.0 + exp(0.0 - v))
      var scale = rate * (y[i] - pred) * pred * (1.0 - pred) * 4.0
      w0 += x0 * scale
      w1 += x1 * scale
      w2 += x2 * scale
      w3 += x3 * scale
    endfor
  endfor
  return [w0, w1, w2, w3]
enddef

def MakeVocab(names: list<string>): dict<float>
  var ns: dict<float> = {}
  for name in names
    if !has_key(ns, name)
      ns[name] = 0.0 + len(ns)
    endif
  endfor
  return ns
enddef

def BagOfWords(names: list<string>, vocab: dict<float>): list<float>
  var l = len(keys(vocab))
  return mapnew(names, (_, val) => vocab[val] / (1.0 * (l - 1)))
enddef

def Shuffle(arr: list<any>): list<any>
  var i = len(arr)
  var j = 0
  while i > 0
    i -= 1
    j = float2nr(rand() / 4294967295.0 * i) % len(arr)
    if i ==# j
      continue
    endif
    [arr[i], arr[j]] = [arr[j], arr[i]]
  endwhile
  return arr
enddef

def Token(line: string): list<any>
  var tok: list<any> = split(line, ',')
  return mapnew(tok[: 3], (_, val) => str2float(val)) + tok[4 :]
enddef

def Main()
  var data: list<any> = mapnew(readfile('iris.csv')[1 :], (_, line) => Token(line))
  call Shuffle(data)
  var train = data[: len(data) / 2]
  var test = data[len(data) / 2 + 1 :]

  var X: list<list<float>> = []
  var y: list<string> = []
  for row in train
    call add(X, row[: 3])
    call add(y, row[4])
  endfor
  var vocab = MakeVocab(y)
  var Y = BagOfWords(y, vocab)
  var ni = mapnew(sort(mapnew(keys(vocab), (_, val) => [val, float2nr(vocab[val])]), (a, b) => a[1] - b[1]), 'v:val[0]')
  var w = LogisticRegression(X, Y, 0.01, 5000)

  var count = 0
  var size = (len(vocab) - 1)
  for row in test
    var r = Softmax(row[: 3], w)
    if ni[min([float2nr(r * size + 0.1), size])] ==# row[4]
      count += 1
    endif
  endfor
  echo (0.0 + count) / (0.0 + len(test))
enddef

def Benchmark()
  var start = reltime()
  call Main()
  echomsg str2float(reltimestr(reltime(start)))
enddef

call Benchmark()
