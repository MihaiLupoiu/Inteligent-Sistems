#pragma config(Sensor, S1,     touchSensor,              sensorTouch)
#pragma config(Sensor, S3,     compassSensor,            sensorVirtualCompass)
#pragma config(Sensor, S2,     lightSensor,              sensorLightActive)
#pragma config(Sensor, S4,     sonarSensor,              sensorSONAR)
#pragma config(Motor,  motorA,          gripperMotor,       tmotorNormal, PIDControl, encoder)
#pragma config(Motor,  motorB,          rightMotor,         tmotorNormal, PIDControl, encoder)
#pragma config(Motor,  motorC,          leftMotor,          tmotorNormal, PIDControl, encoder)
//*!!Code automatically generated by 'ROBOTC' configuration wizard               !!*//
// This is for the NXT model, not TETRIX

/************************************\
|*  ROBOTC Virtual World            *|
|*                                  *|
|*  DO NOT OVERWRITE THIS FILE      *|
|*  MAKE SURE TO "SAVE AS" INSTEAD  *|
\************************************/

task line(){
	while(true){
		if(SensorValue(lightSensor) < 50){
			motor[motorC] = 0;
  		motor[motorB] = 0;
			wait1Msec(1000);

  		motor[motorC] = -75;
   		motor[motorB] = -75;
   		wait1Msec(1000);

    	int valor = random(2);
  		if(valor == 1){
  			motor[motorC] = 0;
     		motor[motorB] = 75;
     		wait1Msec(500);
    	}else{
    		motor[motorC] = 75;
     		motor[motorB] = 0;
     		wait1Msec(500);
			}
		}
	}
}

task main(){
		StartTask(line);
	while(true){
		motor[motorC] = 75;
    motor[motorB] = 75;
    wait1Msec(random(50)*100);

    int valor = random(2);
  	if(valor == 1){
  		motor[motorC] = 0;
   		motor[motorB] = 75;
   		wait1Msec(random(20)*100);
    }else{
   		motor[motorC] = 75;
   		motor[motorB] = 0;
   		wait1Msec(random(20)*100);
     }
  }
}
