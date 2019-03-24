#!/bin/bash

MOSES_GIT_URL=${MOSES_GIT_URL:-https://github.com/moses-smt/mosesdecoder.git}
SUBWORD_GIT_URL=${SUBWORD_GIT_URL:-https://github.com/rsennrich/subword-nmt.git}
SEQ2SEQ_GIT_URL=${SEQ2SEQ_GIT_URL:-https://github.com/google/seq2seq.git}

MOSES_TOKENIZER=${MOSES_SCRIPT:-mosesdecoder/scripts/tokenizer/tokenizer.perl}
MOSES_CLEAN=${MOSES_CLEAN:-mosesdecoder/scripts/training/clean-corpus-n.perl}
VOCAB_SKRIPT=${VOCAB_SKRIPT:-seq2seq/bin/tools/generate_vocab.py}

LEARN_BPE_SKRIPT=${LEARN_BPE_SKRIPT:-subword-nmt/learn_bpe.py}
APPLY_BPE_SKRIPT=${APPLY_BPE_SKRIPT:-subword-nmt/apply_bpe.py}
GET_VOCAB_SKRIP=${GET_VOCAB_SKRIP:-subword-nmt/get_vocab.py}
NUM_THREADS=${NUM_THREADS:-8}
PERL=${PERL:-perl}
PYTHON=${PYTHON:-python}

SOURCE=corpus.src
TARGET=corpus.dst

cp $1 $SOURCE
cp $2 $TARGET

if [ "$#" -ne 2 ]; then
  echo "Source and target files must be passed"
  exit 1
fi

if [ ! -d "mosesdecoder" ]; then
  git clone ${MOSES_GIT_URL}
fi

if [ ! -d "subword-nmt" ]; then
  git clone ${SUBWORD_GIT_URL}
fi

if [ ! -d "seq2seq" ]; then
  git clone ${SEQ2SEQ_GIT_URL}
fi

# Tokenization >> corpus.src && corpus.dst
cat $SOURCE | ${PERL} ${MOSES_TOKENIZER} -threads ${NUM_THREADS} -l en > $SOURCE.tok
cat $TARGET | ${PERL} ${MOSES_TOKENIZER} -threads ${NUM_THREADS} -l en > $TARGET.tok

mv $SOURCE.tok $SOURCE
mv $TARGET.tok $TARGET

# Cleaning >> corpus.clean.src && corpus.clean.dst
${PERL} ${MOSES_CLEAN} corpus src dst "corpus.clean" 1 250

# Creat Vocabulary src.voc && dst.vo:
${PYTHON} ${VOCAB_SKRIPT} --max_vocab_size 50000 < corpus.clean.src > src.voc
${PYTHON} ${VOCAB_SKRIPT} --max_vocab_size 50000 < corpus.clean.dst > dst.voc

# Learn Shared BPE >> bpe.32000
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "corpus.clean.src" "corpus.clean.dst" | \
    ${PYTHON} ${LEARN_BPE_SKRIPT} -s $merge_ops > "bpe.${merge_ops}"i

# Apply BPE >> corpus.clean.bpe.32000.dst && corpus.clean.bpe.32000.src
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in src dst; do
    for f in "corpus.${lang}" "corpus.clean.${lang}"; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${PYTHON} ${APPLY_BPE_SKRIPT} -c "bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE >> vocab.bpe.32000
  cat "corpus.clean.bpe.32000.src" "corpus.clean.bpe.32000.dst" | \
    ${PYTHON} ${GET_VOCAB_SKRIP} | cut -f1 -d ' ' > "vocab.bpe.${merge_ops}"
done
