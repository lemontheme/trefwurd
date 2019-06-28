import os
import html
import csv
import re
import plac


@plac.annotations(
    elex_txt_f="\\\\-delimited variant of eLex data available in original download",
    out_f="File to write to"
)
def main(elex_txt_f, out_f):
    f = os.path.expanduser(elex_txt_f)
    f_encoding = "ascii"
    pos_tag_regex = re.compile(r"^(\w+)\((.*)\)$")
    result = []
    with open(f, encoding=f_encoding) as fp:
        escaped_fp = (html.unescape(line) for line in fp)
        reader = csv.reader(escaped_fp, delimiter="\\")
        for row in reader:
            lemma, word, pos, freq = row[1], row[8], row[9], row[17]
            if word.isnumeric():  # filter out numbers
                continue
            m = re.match(pos_tag_regex, pos)
            tag, morph = m.groups()
            pos = tag
            result.append((word, lemma, pos, morph, freq))
    print("result:")
    print(result[:20])

    elex_tsv = out_f
    with open(elex_tsv, 'wt') as fp:
        w = csv.writer(fp, delimiter="\t")
        w.writerows(result)
    print("Done!")


if __name__ == '__main__':
    plac.call(main)
