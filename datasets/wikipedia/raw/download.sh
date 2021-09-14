#!/bin/bash

for lang in af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh
do
    echo "==================================================="
    echo "$lang"
    echo ""
    wget https://dumps.wikimedia.org/${lang}wiki/latest/${lang}wiki-latest-pages-articles.xml.bz2
    bzip2 -d ${lang}wiki-latest-pages-articles.xml.bz2
    python -m wikiextractor.WikiExtractor --processes 1 --output ${lang} ${lang}wiki-latest-pages-articles.xml
    rm ${lang}wiki-latest-pages-articles.xml
    echo ""
done
