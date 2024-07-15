import geoTherm as gt


## Water Circuit
water = 'H2O'


# Set the default input/output units 
gt.units.input = 'SI'
gt.units.output = 'mixed'


# Hot Circuit
HOT = gt.Model([gt.Boundary(name='Well', fluid=water, P=(40, 'bar'), T=473),
              gt.HEX(name='WaterHEX', US='Well', DS='WaterHEXOut', dP =(38, 'bar'),w=50.232),
              gt.TBoundary(name='WaterHEXOut', fluid=water, P=(2, 'bar'), T=314.31038),
              gt.POutlet(name='Outlet', fluid=water, P=(140, 'bar'), T=500),
              gt.fixedWPump(name='WaterPump',eta=.7,PR=1,w=(90,'kg/s'),US='WaterHEXOut',DS='Outlet')])



# Solve Hot
HOT.solve()

## ORC Circuit
fluid = 'isobutane'

ww = 50
ORC = gt.Model([gt.Boundary(name='LowT', fluid=fluid, P=(5.1180101718524, 'bar'), T=308),
              gt.Pump(name='Pump', eta=.7, PR=30/5.1180101718524, w=50, US='LowT', DS='PumpOut'),
              gt.Station(name='PumpOut', fluid=fluid),
              #gt.Qdot(name='ORC_Qdot', hot='WaterHEX'),
              gt.HEX(name='ORC_HEX', US = 'PumpOut', DS = 'TurbIn', w=ww, Q=-HOT['WaterHEX']._Q, dP=(1,'bar'), D=(2, 'in'), L=3),
              gt.Station(name='TurbIn', fluid=fluid),
              gt.fixedWTurbine(name='Turb', eta=.75, PR=1.3, w=ww, US='TurbIn', DS='TurbOut'),
              gt.Station(name='TurbOut', fluid=fluid),
              gt.HEX(name='CoolHex', US = 'TurbOut', DS = 'LowT', w=ww, dP=(1,'bar'))])


# Create empty model
CombinedModel = gt.Model()
# Add Hot
CombinedModel += HOT
# Add ORC
CombinedModel += ORC


CombinedModel += gt.wBalance('TurbInT','Turb','Well.T','TurbIn.T',5)
CombinedModel += gt.TBalance('Blah','WaterHEXOut','WaterHEXOut.T','PumpOut.T',5)

CombinedModel.solve()


# ADD Heat Exchange between HOT and ORC
#CombinedModel += gt.HEXConnector(name='HOT_HEX', hot='WaterHEX', cool='ORC_HEX')


print('HOT WATER Circut')
print(HOT)
print('ORC Circuit')
print(ORC)
print('Combined Thermo')
print(CombinedModel)
