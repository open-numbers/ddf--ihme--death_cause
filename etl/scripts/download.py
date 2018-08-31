# -*- coding: utf-8 -*-

from ddf_utils.factory import ihme as m


def main():
    md = m.load_metadata()
    version = 227
    context = ['cause', 'risk', 'etiology', 'impairment']
    m.bulk_download('../source', version, context)


if __name__ == '__main__':
    main()
