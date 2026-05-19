allowed_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_{}'
p = 'your text here'

p = ''.join(ch for ch in p if ch in allowed_chars)

print(p)
