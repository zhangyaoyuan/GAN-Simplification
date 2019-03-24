#!/bin/bash

MOSES_GIT_URL=${MOSES_GIT_URL:-https://github.com/moses-smt/mosesdecoder.git}
SUBWORD_GIT_URL=${SUBWORD_GIT_URL:-https://github.com/rsennrich/subword-nmt.git}
SEQ2SEQ_GIT_URL=${SEQ2SEQ_GIT_URL:-https://github.com/google/seq2seq.git}

MOSES_TOKENIZER=${MOSES_SCRIPT:-mosesdecoder/scripts/tokenizer/tokenizer.perl}
MOSES_CLEAN=${MOSES_CLEAN:-mosesdecoder/scripts/training/clean-corpus-n.perl}
VOCAB_SKRIPT1=${VOCAB_SKRIPT:-seq2seq/bin/tools/generate_vocab.py}
VOCAB_SKRIPT2=${VOCAB_SKRIPT:-vocab.py}

LEARN_BPE_SKRIPT=${LEARN_BPE_SKRIPT:-subword-nmt/learn_bpe.py}
APPLY_BPE_SKRIPT=${APPLY_BPE_SKRIPT:-subword-nmt/apply_bpe.py}
GET_VOCAB_SKRIP=${GET_VOCAB_SKRIP:-subword-nmt/get_vocab.py}
NUM_THREADS=${NUM_THREADS:-8}
PERL=${PERL:-perl}
PYTHON=${PYTHON:-python}

SOURCE=tune.8turkers3.src
TARGET=tune.8turkers3.dst
CORPUS=tune.8turkers3
CORPUS_CLEAN=tune.8turkers3.clean

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
${PERL} ${MOSES_CLEAN} $CORPUS src dst $CORPUS_CLEAN 1 250




# Learn in Train>> bpe.32000
# Apply BPE >> corpus.clean.bpe.32000.dst && corpus.clean.bpe.32000.src
for merge_ops in 32000; do
  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in src dst; do
    for f in "${CORPUS_CLEAN}.${lang}"; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${PYTHON} ${APPLY_BPE_SKRIPT} -c "bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done
done
