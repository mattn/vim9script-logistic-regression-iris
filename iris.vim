vim9script
def Dot(x: list<float>, y: list<float>): float
  var r = 0.0
  for v in range(len(x))
    r += x[v] * y[v]
  endfor
  return r
enddef

def Scale(x: list<float>, f: float): list<float>
  return map(copy(x), {_, v -> v * f})
enddef

def Add(x: list<float>, y: list<float>): list<float>
  return map(copy(x), {i, v -> v + y[i]})
enddef

def Softmax(w: list<float>, x: list<float>): float
  return 1.0 / (1.0 + exp(0.0 - Dot(w, x)))
enddef

def LogisticRegression(X: list<list<float>>, y: list<float>, rate: float, ntrains: number): list<float>
  var l = 0.0 + len(X[0])
  var w = map(repeat([[]], len(X[0])), {v -> (rand() / 4294967295.0 - 0.5) * l / 2})
  for n in range(ntrains)
    for i in range(len(X))
      var x = X[i]
      var pred = Softmax(x, w)
      var perr = y[i] - pred
      var scale = rate * perr * pred * (1.0 - pred)
      var dx = copy(x)
      dx = Scale(copy(x), scale)
      for j in range(len(x))
        w = Add(w, dx)
      endfor
    endfor
  endfor
  return w
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
  return map(names, {_, val -> vocab[val] / (1.0 * (l - 1))})
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
  return map(tok[:3], {_, val -> str2float(val)}) + tok[4:]
enddef

def Main()
  var data: list<any> = map(readfile('iris.csv'), {_, line -> Token(line)})
  call Shuffle(data)
  var train = data[:100]
  var test = data[101:]

  var X: list<list<float>> = []
  var y: list<string> = []
  for row in train
    call add(X, row[:3])
    call add(y, row[4])
  endfor
  var vocab = MakeVocab(y)
  var Y = BagOfWords(y, vocab)
  var ni = map(sort(map(keys(vocab), {_, val -> [val, float2nr(vocab[val])]}), {a, b -> a[1] - b[1]}), 'v:val[0]')
  var w = LogisticRegression(X, Y, 0.1, 10000)

  var count = 0
  var size = (len(vocab) - 1)
  for row in test
    var r = Softmax(row[:3], w)
    if ni[min([float2nr(r * size + 0.1), size])] ==# row[4]
      count += 1
    endif
  endfor
  echo (0.0 + count) / (0.0 + len(test))
enddef

call Main()
