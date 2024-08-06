from mininet.net import Mininet
from mininet.node import OVSController, OVSSwitch, Node
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

def myNetwork():
    net = Mininet(topo=None, build=False, link=TCLink)

    info('*** Adding controller\n')
    net.addController(name='c0', controller=OVSController, protocol='tcp', port=6633)

    info('*** Add switches\n')
    s1 = net.addSwitch('s1', cls=OVSSwitch)
    s2 = net.addSwitch('s2', cls=OVSSwitch)

    info('*** Add hosts\n')
    h1 = net.addHost('h1', ip='10.0.0.1')
    h2 = net.addHost('h2', ip='10.0.0.2')
    h3 = net.addHost('h3', ip='10.0.0.3')
    h4 = net.addHost('h4', ip='10.0.0.4')
    h5 = net.addHost('h5', ip='10.0.0.5')
    h6 = net.addHost('h6', ip='10.0.0.6')
    h7 = net.addHost('h7', ip='10.0.0.7')
    querier = net.addHost('querier', ip='10.0.0.8')

    info('*** Add links\n')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)
    net.addLink(h4, s2)
    net.addLink(h5, s2)
    net.addLink(h6, s2)
    net.addLink(h7, s2)
    net.addLink(s1, s2)
    net.addLink(querier, s1)

    info('*** Add NAT\n')
    nat = net.addHost('nat0', cls=Node, ip='10.0.0.254', inNamespace=False)
    net.addLink(nat, s1)

    net.start()

    # Setup NAT
    info('*** Setting up NAT\n')
    nat.cmd('ip addr add 10.0.0.254/24 dev nat0-eth0')
    nat.cmd('ip link set nat0-eth0 up')
    nat.cmd('echo 1 > /proc/sys/net/ipv4/ip_forward')
    nat.cmd('iptables -t nat -A POSTROUTING -s 10.0.0.0/24 ! -d 10.0.0.0/24 -j MASQUERADE')
    nat.cmd('iptables -A FORWARD -s 10.0.0.0/24 -j ACCEPT')
    nat.cmd('iptables -A FORWARD -d 10.0.0.0/24 -j ACCEPT')

    # Setup default routes and DNS for hosts
    for host in [h1, h2, h3, h4, h5, h6, h7, querier]:
        host.cmd('ip route add default via 10.0.0.254')
        host.cmd('echo "nameserver 8.8.8.8" > /etc/resolv.conf')

    CLI(net)
    net.stop()

if __name__ == '__main__':
    setLogLevel('info')
    myNetwork()
