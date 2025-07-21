
        from mininet.net import Mininet
        from mininet.cli import CLI
        from mininet.link import TCLink
        from mininet.node import RemoteController
        from linearTopo import MyTopo

        def send():
            topo = MyTopo()
            net = Mininet(topo=topo, controller=RemoteController, link=TCLink)
            net.start()

            h1, h2 = net.get('h1'), net.get('h2')
            print(h1.cmd('ping -c 3 10.0.0.2'))

            if False:
                print(h1.cmd('hping3 -S 10.0.0.2 -p 80 -c 100'))
            else:
                print(h1.cmd('ping -c 10 10.0.0.2'))

            net.stop()

        if __name__ == "__main__":
            send()
        