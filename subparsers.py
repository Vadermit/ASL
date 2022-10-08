import argparse

def add_ml100k_subparser(subparsers):
    subparser = subparsers.add_parser('ml-100k')
    subparser.add_argument('--epochs', type=int, default=3)
    subparser.add_argument('--data', type=str, default='ml-100k')
    subparser.add_argument('--lr_str', type=float, default=5e-3)
    subparser.add_argument('--beta', type=float, default=0.999)
    subparser.add_argument('--loss_weight_reg', type=float, default=0.01)

def add_ml1m_subparser(subparsers):
    subparser = subparsers.add_parser('ml-1m')
    subparser.add_argument('--epochs', type=int, default=5)
    subparser.add_argument('--data', type=str, default='ml-1m')
    subparser.add_argument('--lr_str', type=float, default=1e-4)
    subparser.add_argument('--beta', type=float, default=0.999)
    subparser.add_argument('--loss_weight_reg', type=float, default=0.05)

def add_beauty_subparser(subparsers):
    subparser = subparsers.add_parser('beauty')
    subparser.add_argument('--epochs', type=int, default=5)
    subparser.add_argument('--data', type=str, default='beauty')
    subparser.add_argument('--lr_str', type=float, default=1e-3)
    subparser.add_argument('--beta', type=float, default=0.9999)
    subparser.add_argument('--loss_weight_reg', type=float, default=0)

def add_office_subparser(subparsers):
    subparser = subparsers.add_parser('office-products')
    subparser.add_argument('--epochs', type=int, default=5)
    subparser.add_argument('--data', type=str, default='office-products')
    subparser.add_argument('--lr_str', type=float, default=1e-3)
    subparser.add_argument('--beta', type=float, default=0.999)
    subparser.add_argument('--loss_weight_reg', type=float, default=1e-05)