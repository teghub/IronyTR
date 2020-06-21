

# information about files

--- data ---
'lemmatized-whole.txt': sentences, tokenized and lemmatized.
'binary-whole.txt': binary values of 'lemmatized-whole.txt'.
'lemmatized-<ironic,nonironic>'.txt: 'lemmatized-whole.txt' separated by class.

--- features ---
'basic-features-whole.txt': basic features extracted from 'lemmatized-whole.txt'.
'basic-features-<ironic,nonironic>.txt': 'basic-features-whole.txt' separated by class.

--- fields for feature files in order ---
class: 0-1 for nonironic-ironic.
word-to-token: count of words to count of tokens ratio.
punctuation-to-token: count of punctuation marks to count tokens ratio.
'!'-exists: 0-1
'!'-to-token: '!' count to token count ratio.
'?'-exists: 0-1
'?'-to-token: '?' count to token count ratio.
'(!)'-exists: 0-1
'(!)'-to-token: '(!)' count to token count ratio.
'(?)'-exists: 0-1
'(?)'-to-token: '(?)' count to token count ratio.
'"'-exists: 0-1
'"'-to-token: '"' count to token count ratio.
'...'-exists: 0-1
'...'-to-token: '...' count to token count ratio.
emoticon/emoji-exists: 0-1
emoticon/emoji-to-token: emoticon/emoji count to token count ratio.
interject_exists: 0-1, 1 if any interjection* exists.
repetition_exists: 0-1, 1 if any symbol repeats.
booster_exists: 0-1, 1 if any booster** exists.
caps_exists: 0-1, 1 if any token is written in caps.


*interjections:
bravo
ay
be
aman
ah
aferin
çüş
eee
eh
ey
eyvah
ha
hadi
hey
la
lan
of
oy
oley
oh
öf
tüh
ulan
vay
ya
yazık
yuh

**boosters:
asla
bazen
çok
elbette
en
gerçekten
kesinlikle
keşke
mutlaka
özellikle
rahatlıkla
tamamen
gayet
cik
aynen
iyice
bile





