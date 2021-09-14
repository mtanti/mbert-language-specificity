'''
A set of standard languages extracted from XTREME to limit the variation of languages to 32.
There are language codes for ISO_639-1 and IS_639-3.
'''

ISO_639_1 = set((
    'af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr '
    'ur vi yo zh'
).split(' '))

ISO_639_3 = set((
    'afr ara bul deu ell eng spa est eus fas fin fra heb hin hun ind ita jpn kaz kor mar nld por '
    'rus tam tel tha tgl tur urd vie yor zho'
).split(' '))
