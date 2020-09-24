vim9script
def Dot(x: list<float>, y: list<float>): float
  let r = 0.0
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
  let l = 0.0 + len(X[0])
  let w = map(repeat([[]], len(X[0])), {v -> (rand() / 4294967295.0 - 0.5) * l})
  for n in range(ntrains)
    for i in range(len(X))
      let x = X[i]
      let pred = Softmax(x, w)
      let perr = y[i] - pred
      let scale = rate * perr * pred * (1.0 - pred)
      w = Add(w, Scale(Add(x, x), scale))
    endfor
  endfor
  return w
enddef

def MakeVocab(names: list<string>): dict<float>
  let ns: dict<float> = {}
  for name in names
    if !has_key(ns, name)
      ns[name] = 0.0 + len(ns)
    endif
  endfor
  return ns
enddef

def BagOfWords(names: list<string>, vocab: dict<float>): list<float>
  let l = len(keys(vocab))
  return map(names, {_, val -> vocab[val] / (1.0 * (l - 1))})
enddef

def Shuffle(arr: list<any>): list<any>
  let i = len(arr)
  let j = 0
  while i
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
  let tok: list<any> = split(line, ',')
  return map(tok[:3], {_, val -> str2float(val)}) + tok[4:]
enddef

def Main()
  let data: list<any> = map(readfile('iris.csv'), {_, line -> Token(line)})
  call Shuffle(data)
  let train = data[:100]
  let test = data[101:]

  let X: list<list<float>> = []
  let y: list<string> = []
  for row in train
    call add(X, row[:3])
    call add(y, row[4])
  endfor
  let vocab = MakeVocab(y)
  let Y = BagOfWords(y, vocab)
  let ni = map(sort(map(keys(vocab), {_, val -> [val, float2nr(vocab[val])]}), {a, b -> a[1] - b[1]}), 'v:val[0]')
  let w = LogisticRegression(X, Y, 0.1, 8000)

  let count = 0
  let size = (len(vocab) - 1)
  for row in test
    let r = Softmax(row[:3], w)
    if ni[min([float2nr(r * size + 0.1), size])] ==# row[4]
      count += 1
    endif
  endfor
  echo (0.0 + count) / (0.0 + len(test))
enddef

call Main()
